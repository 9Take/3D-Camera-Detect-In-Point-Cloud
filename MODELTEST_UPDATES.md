# ModelTest.py Enhancement - Multi-Version Template Detection

## Overview
Updated `realsensepy/src/modeltest.py` to automatically load and test multiple template versions from the Model1 folder and use the best-matching template for each target.

## Key Features Added

### 1. **Multi-Version Template Loading**
   - Scans Model1 folder for all template variations (e.g., A.1, A.2, A.3, B.1, B.2, etc.)
   - Expects files in format: `{NAME}_{VERSION}_data.txt` and `{NAME}_{VERSION}_offset.txt`
   - Example: `A.1_data.txt`, `A.1_offset.txt`, `B.2_data.txt`, `B.2_offset.txt`

### 2. **Automatic Best Match Selection**
   - Tests all versions of each template against the camera frame
   - Compares template matching confidence scores
   - Automatically selects the version with highest confidence (≥ 70%)
   - Displays which version was used and its confidence level

### 3. **Enhanced Output**
   - Saves which template version was used for each target
   - Records matching confidence for quality verification
   - Improved logging showing:
     - Confidence % for each template version tested
     - Which version provided the best match
     - Total targets found and their versions

## File Format Expected

The Model1 folder should contain pairs of files:

```
Model1/
├── A.1_data.txt       (template image data as comma/space-separated values)
├── A.1_offset.txt     (2D offset coordinates: x,y)
├── A.2_data.txt
├── A.2_offset.txt
├── A.3_data.txt
├── A.3_offset.txt
├── A.4_data.txt
├── A.4_offset.txt
├── A.5_data.txt
├── A.5_offset.txt
├── B.1_data.txt
├── B.1_offset.txt
├── B.2_data.txt
├── B.2_offset.txt
├── B.3_data.txt
├── B.3_offset.txt
├── B.4_data.txt
└── B.4_offset.txt
```

## New Functions

### `load_all_template_versions(template_dir)`
- **Purpose**: Load all template variations from a directory
- **Input**: Path to template directory (default: Model1)
- **Output**: Dictionary mapping target names to list of (template, offset, version_name)
- **Features**:
  - Loads template data from `.txt` files using numpy
  - Handles both comma-separated and space-separated formats
  - Graceful error handling for corrupted/missing files
  - Detailed logging of loaded templates

### Updated `main()`
- Replaced single-template-per-target with multi-version testing
- Tests all available versions during detection loop
- Tracks best match for each target
- Enhanced 6-DOF output includes template version used

## Usage

### Default (uses Model1 folder):
```bash
python realsensepy/src/modeltest.py
```

### Custom template directory:
```bash
python realsensepy/src/modeltest.py -td /path/to/templates
```

### Specify output directory:
```bash
python realsensepy/src/modeltest.py -td Model1 -sd my_output_folder
```

## Output Data

Generated files include template version information:
```
target_A_data.txt:
Target: A
Template_Version: A.3
Confidence: 85.2%
Position_X: 0.123456
Position_Y: 0.234567
Position_Z: 0.345678
Roll: 12.34
Pitch: 23.45
Yaw: 34.56
```

## How It Works

1. **Initialization**: Loads all template versions from Model1
2. **Detection Loop**: For each frame:
   - Tests each target's template versions
   - Compares matching confidence scores
   - Uses version with highest score
   - Displays real-time confidence % for each version
3. **Capture**: When 'q' is pressed:
   - Captures 100 frames for depth averaging
   - Generates 3D point cloud
   - Extracts 6-DOF for each detected target
   - Saves results with version info

## Benefits

✅ **Robustness**: Multiple template versions = more reliable detection  
✅ **Flexibility**: Can test different template qualities in real-time  
✅ **Automatic**: No need to manually select best version  
✅ **Traceability**: Records which version was used  
✅ **Easy to expand**: Simply add more numbered versions to Model1  

## Dependencies

Same as original, plus:
- `from collections import defaultdict` (standard library)

## Notes

- Templates should be grayscale images stored as 2D arrays
- Both `.txt` and `.npy` file formats supported
- Confidence threshold set to 0.70 (70%)
- All versions logged even if they fail to meet threshold
