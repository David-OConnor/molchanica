# This file sets up a Linux desktop entry, and moves the application into ~/molchanica.
#
# That one directory is the whole install: Molchanica resolves everything it writes relative to
# its own executable, so the preferences file, the cache of molecules downloaded from the web, the
# graphics pipeline cache, and the optional third-party tools installed from the Tools panel all
# end up beside the binary, rather than scattered across ~/.local/share and friends.

NAME_UPPER="Molchanica"
NAME="molchanica"
mpnn_converter="convert_mpnn_weights.py"

APP_DIR="$HOME/${NAME}"
DESKTOP_PATH="$HOME/.local/share/applications/${NAME}.desktop"

chmod +x $NAME

if [ ! -d "$APP_DIR" ]; then
  mkdir "$APP_DIR"
fi

cp "$NAME" "$APP_DIR"
cp icon.png "$APP_DIR/icon.png"
if [ -f "$mpnn_converter" ]; then
  cp "$mpnn_converter" "$APP_DIR/$mpnn_converter"
else
  printf "Warning: %s was not found; native ProteinMPNN ΔΔG conversion will be skipped.\n" \
    "$mpnn_converter"
fi

# We create a .desktop file dynamically here; one fewer file to manage.
cat > "$DESKTOP_PATH" <<EOF
[Desktop Entry]
Name=${NAME_UPPER}
Exec=${APP_DIR}/${NAME}
Icon=${APP_DIR}/icon.png
# Molchanica finds its data next to its executable rather than through the working directory. Path
# is set only so that file dialogs open somewhere predictable.
Path=${APP_DIR}
Type=Application
Terminal=false
Categories=Development;Science;Biology;
Comment=Molecule and protein viewer
EOF

chmod +x "$DESKTOP_PATH"

# If the cuda FFT lib is packaged with the download, place it beside the executable. Molchanica
# loads cuFFT at runtime and looks in its own directory, so this needs neither root nor a
# system-wide install. Machines that already have CUDA use their own copy, and machines without an
# Nvidia driver ignore it and fall back to the CPU.
cufft_lib="libcufft.so.12"
if [ -f "./$cufft_lib" ]; then
  cp "./$cufft_lib" "$APP_DIR/$cufft_lib"
  printf "Copied %s (the cuFFT dependency) to %s.\n" "$cufft_lib" "$APP_DIR"
fi

#read -p "Install gemmi from apt, to support unprocessed electron density files? [y/n] " ans
#if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
#  sudo apt install gemmi
#  printf "\ngemmi installed. You can uninstall it with sudo apt remove gemmi.\n"
#fi

# Optional third-party tools are installed in-process from Molchanica's Tools panel. Each recipe
# keeps its environment and assets under $APP_DIR/process_executables, along with everything else.

printf "\nMoved the ${NAME_UPPER} executable and icon to ${APP_DIR}."
printf "\n\nEverything ${NAME_UPPER} writes stays in that one folder:"
printf "\n  molchanica_prefs.mca    preferences and per-molecule settings"
printf "\n  managed_molecules/      molecules downloaded or generated in the app"
printf "\n  gpu_cache/              the graphics pipeline cache"
printf "\n  process_executables/    optional third-party tools from the Tools panel"
printf "\n\nSet MOLCHANICA_DATA_DIR to keep them somewhere else, e.g. on another drive."
printf "\n\nYou can launch ${NAME_UPPER} through the GUI (e.g., search \"${NAME_UPPER}\") and/or add it to favorites.\n"
