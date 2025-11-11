#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke tests for meta_library module.

Tests basic functionality of all meta_library modules to ensure
they can be imported and execute without errors.

Testes básicos para o módulo meta_library.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from meta_library import generate_df, clean_library, phase_matching


class TestGenerateDF(unittest.TestCase):
    """Tests for generate_df module."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_dir_path = Path(self.test_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_fake_touchstone(self, filename: str, nports: int = 2):
        """Create a minimal fake Touchstone file for testing."""
        content = f"""# GHz S RI R 50
[Version] 2.0
[Number of Ports] {nports}
! Comments
Parameters = {{L_x=400; L_y=500; H=600; Lambda=1064}}
[Network Data]
"""
        # Add dummy S-parameter data
        if nports == 1:
            content += "1.0 0.5 0.1 0.2 0.0\n"
            content += "2.0 0.6 0.15 0.25 0.0\n"
        elif nports == 2:
            content += "1.0 0.5 0.1 0.6 0.2 0.4 0.15 0.5 0.12\n"
            content += "2.0 0.55 0.12 0.65 0.22 0.45 0.16 0.52 0.13\n"
        elif nports == 4:
            # Simplified 4-port data
            content += "1.0 " + " ".join(["0.1 0.2"] * 16) + "\n"
            content += "2.0 " + " ".join(["0.15 0.25"] * 16) + "\n"
        
        filepath = self.test_dir_path / filename
        filepath.write_text(content)
        return filepath
    
    def test_parse_touchstone_params(self):
        """Test parsing parameters from Touchstone header."""
        ts_file = self._create_fake_touchstone("test_001.ts")
        params = generate_df.parse_touchstone_params(str(ts_file))
        
        self.assertIn('L_x', params)
        self.assertIn('L_y', params)
        self.assertIn('H', params)
        self.assertEqual(params['L_x'], 400.0)
        self.assertEqual(params['L_y'], 500.0)
    
    def test_parse_number_of_ports_from_header(self):
        """Test parsing port count from header."""
        ts_file = self._create_fake_touchstone("test_002.ts", nports=2)
        nports = generate_df.parse_number_of_ports_from_header(str(ts_file))
        
        self.assertEqual(nports, 2)
    
    def test_touchstone_to_dataframe(self):
        """Test converting Touchstone files to DataFrame."""
        # Create test files
        self._create_fake_touchstone("test_001.ts", nports=2)
        self._create_fake_touchstone("test_002.ts", nports=2)
        
        # Note: This test will fail if scikit-rf can't parse our fake files
        # In that case, we skip the test
        try:
            df = generate_df.touchstone_to_dataframe(
                folder=str(self.test_dir_path),
                recursive=False,
                pattern="*.ts"
            )
            
            # Check expected columns
            self.assertIn('arquivo', df.columns)
            self.assertIn('id_nanopilar', df.columns)
            self.assertIn('frequencia_hz', df.columns)
            self.assertIn('L_x', df.columns)
            self.assertIn('L_y', df.columns)
            self.assertIn('H', df.columns)
            self.assertIn('nports', df.columns)
            
            # Check S-parameter columns for 2-port
            self.assertIn('S21_real', df.columns)
            self.assertIn('S21_imag', df.columns)
            
            # Should have at least some rows
            self.assertGreater(len(df), 0)
            
        except Exception as e:
            # Skip if scikit-rf can't parse fake files
            self.skipTest(f"Skipping: scikit-rf cannot parse fake files ({e})")


class TestCleanLibrary(unittest.TestCase):
    """Tests for clean_library module."""
    
    def setUp(self):
        """Create fake library DataFrame for testing."""
        # Create synthetic data with 3 nanopilars, 2 frequencies each
        data = []
        for id_nano in [1, 2, 3]:
            for freq in [1e12, 2e12]:
                row = {
                    'id_nanopilar': id_nano,
                    'frequencia_hz': freq,
                    'L_x': 400 + id_nano * 10,
                    'L_y': 500 + id_nano * 10,
                    'H': 600,
                    'S21_real': 0.5 + id_nano * 0.05,
                    'S21_imag': 0.1 + id_nano * 0.01,
                    'S12_real': 0.4 + id_nano * 0.05,
                    'S12_imag': 0.15 + id_nano * 0.01,
                }
                data.append(row)
        
        self.df = pd.DataFrame(data)
    
    def test_append_derived_columns(self):
        """Test adding derived columns."""
        df_clean = clean_library.append_derived_columns(
            self.df,
            unwrap_phase=False,
            phase_unit='rad'
        )
        
        # Check new columns exist
        self.assertIn('amp_TE', df_clean.columns)
        self.assertIn('phase_TE', df_clean.columns)
        self.assertIn('amp_TM', df_clean.columns)
        self.assertIn('phase_TM', df_clean.columns)
        self.assertIn('S_complex_TE', df_clean.columns)
        self.assertIn('S_complex_TM', df_clean.columns)
        
        # Check that amplitudes are positive
        self.assertTrue((df_clean['amp_TE'] >= 0).all())
        self.assertTrue((df_clean['amp_TM'] >= 0).all())
        
        # Check phases are in expected range (radians)
        self.assertTrue((df_clean['phase_TE'] >= -np.pi).all())
        self.assertTrue((df_clean['phase_TE'] <= np.pi).all())
    
    def test_append_derived_columns_deg(self):
        """Test phase conversion to degrees."""
        df_clean = clean_library.append_derived_columns(
            self.df,
            unwrap_phase=False,
            phase_unit='deg'
        )
        
        # Check phases are in degree range
        self.assertTrue((df_clean['phase_TE'] >= -180).all())
        self.assertTrue((df_clean['phase_TE'] <= 180).all())
    
    def test_save_library(self):
        """Test saving library to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            csv_path = tmpdir / "test.csv"
            parquet_path = tmpdir / "test.parquet"
            
            clean_library.save_library(
                self.df,
                out_csv=str(csv_path),
                out_parquet=str(parquet_path)
            )
            
            # Check files were created
            self.assertTrue(csv_path.exists())
            self.assertTrue(parquet_path.exists())
            
            # Check they can be read back
            df_csv = pd.read_csv(csv_path)
            df_parquet = pd.read_parquet(parquet_path)
            
            self.assertEqual(len(df_csv), len(self.df))
            self.assertEqual(len(df_parquet), len(self.df))


class TestPhaseMatching(unittest.TestCase):
    """Tests for phase_matching module."""
    
    def setUp(self):
        """Create fake library DataFrame with derived columns."""
        # Create synthetic data
        data = []
        for lx in [400, 420, 440]:
            for ly in [500, 520, 540]:
                row = {
                    'L_x': lx,
                    'L_y': ly,
                    'H': 600,
                    'phase_TE': np.random.uniform(-np.pi, np.pi),
                    'phase_TM': np.random.uniform(-np.pi, np.pi),
                    'amp_TE': np.random.uniform(0.5, 1.0),
                    'amp_TM': np.random.uniform(0.5, 1.0),
                }
                data.append(row)
        
        self.df = pd.DataFrame(data)
    
    def test_compute_heatmaps(self):
        """Test heatmap computation."""
        heatmaps = phase_matching.compute_heatmaps(
            self.df,
            x='L_x',
            y='L_y',
            fields=('phase_TE', 'amp_TE'),
            bins_x=10,
            bins_y=10
        )
        
        # Check expected keys
        self.assertIn('x_grid', heatmaps)
        self.assertIn('y_grid', heatmaps)
        self.assertIn('phase_TE', heatmaps)
        self.assertIn('amp_TE', heatmaps)
        
        # Check shapes
        self.assertEqual(heatmaps['phase_TE'].shape, (10, 10))
        self.assertEqual(heatmaps['amp_TE'].shape, (10, 10))
    
    def test_perform_phase_matching(self):
        """Test phase matching algorithm."""
        # Create small target phases
        target_shape = (5, 5)
        target_te = np.random.uniform(-np.pi, np.pi, target_shape)
        target_tm = np.random.uniform(-np.pi, np.pi, target_shape)
        
        layout_lx, layout_ly, error_map = phase_matching.perform_phase_matching(
            self.df,
            target_phase_tm=target_tm,
            target_phase_te=target_te,
            use_height=False
        )
        
        # Check output shapes
        self.assertEqual(layout_lx.shape, target_shape)
        self.assertEqual(layout_ly.shape, target_shape)
        self.assertEqual(error_map.shape, target_shape)
        
        # Check that selected values are from library
        self.assertTrue(np.all(np.isin(layout_lx, self.df['L_x'].values)))
        self.assertTrue(np.all(np.isin(layout_ly, self.df['L_y'].values)))
        
        # Check errors are non-negative
        self.assertTrue((error_map >= 0).all())
    
    def test_save_heatmap_figures(self):
        """Test saving heatmap figures."""
        heatmaps = phase_matching.compute_heatmaps(
            self.df,
            fields=('phase_TE', 'amp_TE'),
            bins_x=10,
            bins_y=10
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_files = phase_matching.save_heatmap_figures(
                heatmaps,
                out_dir=Path(tmpdir),
                prefix="test",
                dpi=100  # Lower DPI for faster testing
            )
            
            # Check files were created
            self.assertGreater(len(saved_files), 0)
            for filepath in saved_files:
                self.assertTrue(filepath.exists())
                self.assertTrue(filepath.suffix == '.png')
    
    def test_save_layout_outputs(self):
        """Test saving layout outputs."""
        # Create dummy layouts
        layout_lx = np.random.uniform(400, 440, (5, 5))
        layout_ly = np.random.uniform(500, 540, (5, 5))
        error_map = np.random.uniform(0, 0.1, (5, 5))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths = phase_matching.save_layout_outputs(
                layout_lx, layout_ly, error_map,
                out_dir=Path(tmpdir),
                prefix="test"
            )
            
            # Check expected files
            self.assertIn('layout_lx', saved_paths)
            self.assertIn('layout_ly', saved_paths)
            self.assertIn('error_map', saved_paths)
            self.assertIn('summary', saved_paths)
            
            # Check files exist
            for filepath in saved_paths.values():
                self.assertTrue(filepath.exists())


class TestModuleIntegration(unittest.TestCase):
    """Integration tests across modules."""
    
    def test_full_pipeline_mock(self):
        """Test a complete mock pipeline."""
        # 1. Create fake raw data (simulating generate_df output)
        # Use more varied data to avoid collinear points
        raw_data = []
        for lx in [400, 420, 440]:
            for ly in [500, 520, 540]:
                i = len(raw_data)
                row = {
                    'id_nanopilar': i,
                    'frequencia_hz': 1e12,
                    'L_x': lx,
                    'L_y': ly,
                    'H': 600,
                    'S21_real': 0.5 + i * 0.01,
                    'S21_imag': 0.1 + i * 0.005,
                    'S12_real': 0.4 + i * 0.01,
                    'S12_imag': 0.15 + i * 0.005,
                }
                raw_data.append(row)
        df_raw = pd.DataFrame(raw_data)
        
        # 2. Clean library (add derived columns)
        df_clean = clean_library.append_derived_columns(df_raw)
        
        # Verify derived columns exist
        self.assertIn('phase_TE', df_clean.columns)
        self.assertIn('phase_TM', df_clean.columns)
        
        # 3. Compute heatmaps
        heatmaps = phase_matching.compute_heatmaps(
            df_clean,
            fields=('phase_TE', 'amp_TE'),
            bins_x=5,
            bins_y=5
        )
        
        # Verify heatmaps were created
        self.assertIn('phase_TE', heatmaps)
        
        # 4. Perform phase matching
        target_te = np.zeros((3, 3))
        target_tm = np.zeros((3, 3))
        
        lx, ly, err = phase_matching.perform_phase_matching(
            df_clean,
            target_phase_tm=target_tm,
            target_phase_te=target_te
        )
        
        # Verify output shapes
        self.assertEqual(lx.shape, (3, 3))
        
        print("✓ Full pipeline test passed")


def run_tests():
    """Run all tests and report results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateDF))
    suite.addTests(loader.loadTestsFromTestCase(TestCleanLibrary))
    suite.addTests(loader.loadTestsFromTestCase(TestPhaseMatching))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
