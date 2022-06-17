# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [unreleased.Features] - YYYY-MM-DD

### Added

### Changed

### Deprecated

### Removed

### Fixed

- Fixed python 3.7 support

## [1.0.0] - 2022-06-16

### Added

- Added the ``elems`` interface to customize datasets

## Changed

- Changed the ``__init__`` interface to initialize raw, depth prediction and depth completion datasets

## [0.2.3] - 2020-11-20

### Fixed

- Fixed intrinsics returned by KittiDepthCompletionDataset


## [0.2.2] - 2020-11-20

### Fixed

- Fixed random state handling in ApplyToFeatures


## [0.2.1] - 2020-11-20

### Fixed

- Fixed transform application in KittiDepthCompletionDataset when using `load_previous` and `load_stereo`
- Fixed transform application in KittiRawDataset when using `load_previous`


## [0.2.0] - 2020-11-19

### Added

- Added KittiRawDataset `download` implementation for 'sync+rect'
- Added KittiRawDataset `load_previous` option
- Added ratio_threshold metric function

### Fixed

- Fixed testing fixture `depth_completion_path` for validation data generation
- Fixed KittiDepthCompletionDataset `load_previous`
- Fixed KittiDepthCompletionDataset on test data loading
- Fixed KittiDepthPredictionDataset `load_previous`
- Fixed KittiDepthPredictionDataset on test data loading

## [0.1.0] - 2020-11-17

### Added

- Added functional metrics rmse, mse, irmse, mae, imar, sq_rel_error, abs_rel_error, silog
- Added ApplyToFeatures, AddFeatures transforms
- Added `torch_kitti` script to download and scaffold datasets
- Added KittiDepthPreictionDataset to load/scaffold Depth Prediction Dataset
- Added KittiDepthCompletionDataset to load/scaffold Depth Completion Dataset
- Added KittiRawDataset to load sync+rect data
