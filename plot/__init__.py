import matplotlib.pyplot as plt
from matplotlib import offsetbox
import basic
from sets import Set
from sklearn.manifold import TSNE
import utils

def save_datasets(in_path,out_path,color_helper=None):
    utils.make_dir(out_path)
    paths=utils.bottom_files(in_path)
    for in_path_i in paths:
        name_i=in_path_i.split('/')[-1]
        out_path_i=out_path+'/'+name_i
        dataset=basic.read_dataset(in_path_i)
        X,y=dataset.X,dataset.y
        color_helper=PersonColors(dataset.persons)
        X=TSNE(n_components=2,perplexity=30).fit_transform(X)
        plt_i=plot_embedding(X,y,title=name_i,color_helper=color_helper,show=False)
        plt.savefig(out_path_i)

def show_dataset(in_path,title="plot"):
    dataset=basic.read_dataset(in_path)
    color_helper=CatColors(np.unique(dataset.y))
    X,y=dataset.X,dataset.y
    X=TSNE(n_components=2,perplexity=30).fit_transform(X)
    print(X.shape)
    plot_embedding(X,y,title=title,color_helper=color_helper,show=True)

def plot_embedding(X,y,title="plot",color_helper=None,show=True):
    n_points=X.shape[0]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
   
    color_helper=color_helper if(color_helper) else lambda i,y_i:0
    plt.figure()
    ax = plt.subplot(111)

    for i in range(n_points):
        color_i= color_helper(i,y[i])
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                   color=plt.cm.tab20( color_i),
                   fontdict={'weight': 'bold', 'size': 9})
    print(x_min,x_max)
    #plt.xticks(np.arange(x_min, x_max, 0.005)), plt.yticks([])
    if title is not None:
        plt.title(title)
    if(show):
        plt.show()
    return plt