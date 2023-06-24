import matplotlib.pyplot as plt
def plot_lists(x_label, y_label, plot1_list, plot1_annotation, plot2_list, plot2_annotation, plot3_list, plot3_annotation):
    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(plot1_list, label=plot1_annotation)
    axs.plot(plot2_list, label=plot2_annotation)
    axs.plot(plot3_list, label=plot3_annotation)
    axs.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
     