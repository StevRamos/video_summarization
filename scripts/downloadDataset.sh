pip3 install gshell==5.5.2
gshell init
gshell cd --with-id 1Am_kJm2VRJo4FfMWls039ZdNJVtkVrPN
gshell download datasets.zip
unzip datasets.zip
rm datasets.zip
unzip CoSum.zip
unzip MVS1K.zip
unzip SumMe.zip -d SumMe
unzip VSUMM.zip
unzip visiocity.zip
mkdir TVsum
tar zxvf tvsum50_ver_1_1.tgz -C TVsum/
rm *.zip
rm *.tgz
