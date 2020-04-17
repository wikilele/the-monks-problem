import matplotlib.pyplot as plt 


class PlotService():

    def plot_distribution(self, distribution):
        x = list(distribution.keys())
        y = list(distribution.values())
        labels = [str(x[0]), str(x[1])]
        plt.bar(x,y, tick_label = labels , width = 0.4, color = ['red', 'green']) 
        
        plt.xlabel('calsses') 
        plt.ylabel('distribution') 
        plt.title('Data Distribution') 
        plt.show() 

    def plot_error(self, epochs, error):
        plt.plot(epochs,error)

        plt.xlabel('epochs') 
        plt.ylabel('error') 
        plt.title('Mean Square Error graph') 
        plt.show() 