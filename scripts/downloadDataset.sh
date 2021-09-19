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
unzip TVsum/ydata-tvsum50-v1_1/ydata-tvsum50-data.zip -d TVsum/ydata-tvsum50-v1_1/
unzip TVsum/ydata-tvsum50-v1_1/ydata-tvsum50-matlab.zip -d TVsum/ydata-tvsum50-v1_1/
unzip TVsum/ydata-tvsum50-v1_1/ydata-tvsum50-thumbnail.zip -d TVsum/ydata-tvsum50-v1_1/
unzip TVsum/ydata-tvsum50-v1_1/ydata-tvsum50-video.zip -d TVsum/ydata-tvsum50-v1_1/
rm TVsum/ydata-tvsum50-v1_1/*.zip
rm *.zip
rm *.tgz
