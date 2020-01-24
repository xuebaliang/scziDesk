Deep soft K-means clustering with self-training for single cell RNA sequence data.
=====
Architecture
-----
![model](https://github.com/xuebaliang/scziDesk/blob/master/Architecture/scziDesk_architecture.JPG)
Requirement
-----
Python 3.6
Tensorflow 1.14
Keras 2.2
Data availability
-----
The real data sets we used can be download in <a href="https://drive.google.com/drive/folders/1BIZxZNbouPtGf_cyu7vM44G5EcbxECeu">data</a>.
Quick start
-----
We use the dataset “Bach” and ZINB distribution modelling to give an example. You just run the following code in your command lines:
python zidpkm.py --dataname "Bach"
Then you will get the cluster result of Bach dataset using scziDesk method in ten random seed. The median values of Accuracy, ARI and NMI are 0.9046, 0.8738 and 0.8343, respectively. 
Help
-----
We are making the whole codes into a python package, and then it will be released. 
