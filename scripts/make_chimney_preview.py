"""Make the README animation from the final MP4 without changing playback speed."""
import subprocess
import sys
from pathlib import Path
import imageio_ffmpeg

source, output = map(Path, sys.argv[1:3])
output.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([
    imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(source),
    "-filter_complex",
    "[0:v]fps=12,scale=320:320:flags=lanczos,split[a][b];"
    "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a",
    "-loop", "0", str(output),
], check=True)
