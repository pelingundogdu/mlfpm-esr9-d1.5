import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_error(out_file,xticks,means,errors):
    print out_file
    print xticks
    print means
    print errors
    x=np.array(range(1,len(xticks)+1))
    my_xticks=list(xticks)
    for index,val in enumerate(xticks):
        print val
        if val.startswith('Dense') or val.startswith('PPITF') :
            my_xticks[index]='NN'+str(index)
        else:
            my_xticks[index]='F'+str(index)
        print my_xticks[index]

    y1=means[1]
    y2=means[2]
    y3=means[3]
    y4=means[4]
    e1=errors[1]
    e2=errors[2]
    e3=errors[3]
    e4=errors[4]
    '''
    x = np.array(range(1,5))
    y1 = np.power(x,0)
    y2 = np.power(x,1)
    y3 = np.power(x,2)
    y4 = np.power(x,3)
    e1 = np.array(1*range(1,5))
    e2 = np.array(2*range(1,5))
    e3 = np.array(3*range(1,5))
    e4 = np.array(4*range(1,5))
    my_xticks=['b','c','d','e']
    '''
    #print x,y,e
    #plt.xticks(x,my_xticks)
    #plt.xlim(0,10)
    #plt.errorbar(x,y,e,linestyle='None',marker='^')
    #plt.show()
    #plt.savefig('kerker.png')

    # Three subplots sharing both x/y axes
    f, (ax1, ax2, ax3,ax4) = plt.subplots(4, sharex=True, sharey=True,figsize=(12,9))
    #ax1.plot(x, y1)
    ax1.errorbar(x,y1,e1,linestyle='None',marker='^')
    ax2.errorbar(x,y2,e2,linestyle='None',marker='^')
    ax3.errorbar(x,y3,e3,linestyle='None',marker='^')
    ax4.errorbar(x,y4,e4,linestyle='None',marker='^')
    ax1.set_title('#testing cell type = 2')
    ax2.set_title('#testing cell type = 4')
    ax3.set_title('#testing cell type = 6')
    ax4.set_title('#testing cell type = 8')
    #ax2.scatter(x, y)
    #ax3.scatter(x, 2 * y ** 2 - 1, color='r')
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    #f.subplots_adjust(hspace=10)
    #plt.figure(figsize=(,10))
    #size=f.get_size_inches()
    #print size

    plt.xlim(0,len(xticks)+1)
    plt.ylim(0,1)
    plt.xticks(x,my_xticks)
    #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    #plt.show()
    plt.savefig(out_file,dpi=300)
    mn_file=open(out_file+'.csv','w')
    for a,b in zip(my_xticks,xticks):
        mn_file.write(a+','+b+"\n")

