import sys
import unittest
from io import StringIO
from get_trajectory import GetInfo

class TestGetInfo(unittest.TestCase):
    """Unit tests for the GetInfo class"""

    def setUp(self):
        """Set up the test case"""
        self.trr_file = 'test.trr'
        self.log_stream = StringIO()  # To capture log messages
        self.get_info = GetInfo(self.trr_file, log=self.log_stream)

    def test_path_existence(self):
        """Test if path_exist method returns correct result"""
        # Test when file exists
        self.assertTrue(self.get_info.path_exist(self.trr_file))

        # Test when file does not exist
        self.assertFalse(self.get_info.path_exist('nonexistent_file.trr'))

    def test_read_traj(self):
        """Test if read_traj method reads the trajectory file"""
        # Replace sys.stdout with a StringIO to capture print output
        stdout_backup = sys.stdout
        sys.stdout = self.log_stream

        self.get_info.read_traj()

        sys.stdout = stdout_backup

        # Assert that the log message indicates successful file reading
        log_output = self.log_stream.getvalue()
        self.assertIn(
            f'Trajectory file `{self.trr_file}` and topology file', log_output)

    def test_get_residues(self):
        """Test if get_residues method returns correct residue indices"""
        expected_residue_indices = {
            'residue1': [0, 1, 2],
            'residue2': [3, 4, 5],
            # Add more expected residue indices as needed
        }

        self.assertEqual(self.get_info.get_residues(),
                         expected_residue_indices)

    def test_get_nums(self):
        """Test if get_nums method returns correct number values"""
        expected_nums = {
            'n_atoms': 100,
            'total_mass': 1000.0,
            'n_frames': 10,
            'totaltime': 100.0,
        }

        self.assertEqual(self.get_info.get_nums(), expected_nums)

if __name__ == '__main__':
    unittest.main()
