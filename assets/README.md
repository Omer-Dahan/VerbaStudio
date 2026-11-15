# Brand assets

`banner.png` and `icon.png` are the shipped brand art for VerbaStudio: the wide banner carried in the README header
and the square app mark. Both come from the owner's brand artwork, resized for the repository
(`1600x533` and `512x512`).

## Logo candidates (kept for reference)
`logo-1.svg` is a three-block caption bar with an accent block and underline.
`logo-2.svg` is a two-line caption lockup with a yellow final segment.
`logo-3.svg` builds the Hebrew letter `כ` from connected caption blocks.
`logo-4.svg` places two caption lines inside a square-cornered caption card.
The palette is `#0B1220`, `#FFFFFF`, and `#FFD400`.
Re-rasterize a source with `ffmpeg -v error -y -i assets/logo-1.svg -frames:v 1 assets/preview-1-512.png` and add `-vf scale=32:32` for the 32 px preview.
Build the contact sheet with `ffmpeg -v error -y -f lavfi -i color=c=0x1A1A1A:s=2288x592 -i assets/preview-1-512.png -i assets/preview-2-512.png -i assets/preview-3-512.png -i assets/preview-4-512.png -filter_complex "[0:v][1:v]overlay=40:40[a];[a][2:v]overlay=652:40[b];[b][3:v]overlay=1264:40[c];[c][4:v]overlay=1876:40" -frames:v 1 assets/logo-candidates.png`.

## Earlier banner draft
The 1600x420 draft (before the brand art above) was drawn with the Heebo family from `fonts/Heebo/`; regenerating it
needs that family installed system-wide because libass and drawtext resolve fonts through fontconfig.
Palette: navy `#0B1220`, white `#FFFFFF`, yellow `#FFD400`, and tagline `#C7D2E0`.
```sh
ffmpeg -v error -y -f lavfi -i "color=c=0x0B1220:s=1600x420:d=1" -i mark.png -filter_complex "[1:v]scale=320:320[m];[0:v][m]overlay=70:50[bg];[bg]drawbox=x=480:y=252:w=300:h=7:color=0xFFD400:t=fill[b1];[b1]drawtext=fontfile=fonts/Heebo/Heebo-Black.ttf:text='VerbaStudio':fontsize=136:fontcolor=white:x=470:y=96[d1];[d1]drawtext=fontfile=fonts/Heebo/Heebo-Regular.ttf:text='From speech to subtitles. All on your device.':fontsize=46:fontcolor=0xC7D2E0:x=480:y=292" -frames:v 1 -update 1 banner.png
```
