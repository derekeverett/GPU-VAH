rm -R output
mkdir output
rm gpu-vah
make clean
make
./gpu-vah --config rhic-conf/ -o output -h
