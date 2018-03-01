mkdir datasets
FILE=tiny-imagenet-200
URL=http://cs231n.stanford.edu/tiny-imagenet-200.zip
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
