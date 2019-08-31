# Face reconstruction and relighting

[Duan Gao](https://gao-duan.github.io/)

| <img src="./data/example/image.png" alt="img" /> | <img src="./res/reconstruct.bmp" alt="img"  /> | <img src="./res/albedo_high.bmp" alt="img"  /> | <img src="./res/relit/out.gif" alt="img"/> | <img src="./res/prog/out.gif" alt="img"/> |
| ------------------------------------------------ | ---------------------------------------------- | ---------------------------------------------- | ------------------------------------------ | ----------------------------------------- |
| <img src="./data/example/me.jpg" alt="img" /> | <img src="./res2/reconstruct.bmp" alt="img"  /> | <img src="./res2/albedo_high.bmp" alt="img"  /> | <img src="./res2/relit/out.gif" alt="img"/> | <img src="./res2/prog/out.gif" alt="img"/> |
| input image                                      | reconstruct image                              | high-frequency albedo                          | relit video                                | relit video(after propagation)            |

Project for reconstructing 3D face from a single image and recovering high-frequency albedo. This can be used in many face applications like face relighting and face editing. You can find more algorithm details in [documentation](./documentation/documentation.pdf).

## Dependencies

- OpenCV 3.4
- Dlib
- Only support Visual Studio now. (Maybe extend to CMake later.)

## Compile

- Set opencv and dlib dependencies inside Visual Studio.
- Download BFM data (https://faces.dmi.unibas.ch/bfm/).
- Convert BFM data to 3 files: bfm_data, bfm_exp and bfm_attrib.txt (see BFM::load() in bfm.cpp for more details of the file format)
- Put BFM data into ./data/BFM/
- Build and run.

## Command line parameters

```./FaceRelighting.exe $SOLUTION_PATH  $OUTPUT_FOLDER  $INPUT_IMAGE_PATH```

> Example:
>
> cd FaceRelighting
>
> PATH_TO_EXE/FaceRelighting.exe   ./  ./res/  ./data/example/image.png




