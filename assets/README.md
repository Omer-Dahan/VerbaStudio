# Logo candidates
`logo-1.svg` is a three-block caption bar with an accent block and underline.
`logo-2.svg` is a two-line caption lockup with a yellow final segment.
`logo-3.svg` builds the Hebrew letter `כ` from connected caption blocks.
`logo-4.svg` places two caption lines inside a square-cornered caption card.
The palette is `#0B1220`, `#FFFFFF`, and `#FFD400`.
Re-rasterize a source with `ffmpeg -v error -y -i assets/logo-1.svg -frames:v 1 assets/preview-1-512.png` and add `-vf scale=32:32` for the 32 px preview.
Build the contact sheet with `ffmpeg -v error -y -f lavfi -i color=c=0x1A1A1A:s=2288x592 -i assets/preview-1-512.png -i assets/preview-2-512.png -i assets/preview-3-512.png -i assets/preview-4-512.png -filter_complex "[0:v][1:v]overlay=40:40[a];[a][2:v]overlay=652:40[b];[b][3:v]overlay=1264:40[c];[c][4:v]overlay=1876:40" -frames:v 1 assets/logo-candidates.png`.
