# This file sets up a Linux desktop entry, and moves the application to the home directory.

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

# Optional third-party tools are installed in-process from Molchanica's Tools panel.
# Each recipe keeps its environment and assets under the per-user Molchanica data directory.

printf "\nMoved the ${NAME_UPPER} executable and icon to ${APP_DIR}."
printf "\n\nYou can launch ${NAME_UPPER} through the GUI (e.g., search \"${NAME_UPPER}\") and/or add it to favorites.\n"
