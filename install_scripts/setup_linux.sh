# This file sets up a Linux desktop entry, and moves the application to the home directory.

NAME_UPPER="Molchanica"
NAME="molchanica"

APP_DIR="$HOME/${NAME}"
DESKTOP_PATH="$HOME/.local/share/applications/${NAME}.desktop"

chmod +x $NAME

if [ ! -d "$APP_DIR" ]; then
  mkdir "$APP_DIR"
fi

cp "$NAME" "$APP_DIR"
cp icon.png "$APP_DIR/icon.png"

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

# If the cuda FFT lib is packaged with the download, move it to the correct place.
cufft_lib="libcufft.so.12"
if [ -f "./$cufft_lib" ]; then
  sudo cp "./$cufft_lib" /usr/lib/
  printf "Moved the libcufft.so.12 library (for the cuFFT dependency)  to /usr/lib.\n"
fi

#read -p "Install gemmi from apt, to support unprocessed electron density files? [y/n] " ans
#if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
#  sudo apt install gemmi
#  printf "\ngemmi installed. You can uninstall it with sudo apt remove gemmi.\n"
#fi

## Each of these installs into its own uv-managed environment under
## ${XDG_DATA_HOME:-$HOME/.local/share}/molchanica, so nothing touches the system Python. The first
## Python-backed tool install also installs uv for the current user if it is not already available.
#read -p "Install OpenDDE, to support structure prediction? Warning: Multi-Gb. [y/n] " ans
#if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
#  ./install_tool.sh opendde
#  printf "\nOpenDDE installed.\n"
#fi

#read -p "Install the antibody tools (IgBLAST and ANARCII)? These are comparatively small. [y/n] " ans
#if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
#  ./install_tool.sh igblast anarcii
#  printf "\nAntibody tools installed.\n"
#fi
#
#printf "\nOther optional tools (boltz2, ligandmpnn, proteinmpnn) can be installed at any time with\n"
#printf "install_scripts/install_tool.sh. Run it with --list to see them all, or 'all' for everything.\n"
#printf "Molchanica's \"Tools\" panel shows which are installed and working.\n"

printf "\nMoved the ${NAME_UPPER} executable and icon to ${APP_DIR}."
printf "\n\nYou can launch ${NAME_UPPER} through the GUI (e.g., search \"${NAME_UPPER}\") and/or add it to favorites.\n"
