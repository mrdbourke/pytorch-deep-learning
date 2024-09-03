# download cifar10
cd ../data
wget https://pjreddie.com/media/files/cifar.tgz
tar xzf cifar.tgz

# extract_cifar_labels
cat cifar/labels.txt

# generate file+paths for training, testing
cd cifar
find `pwd`/train -name \*.png > train.list
find `pwd`/test -name \*.png > test.list
cd ../..