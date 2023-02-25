"""
This code defines a function plot that uses the matplotlib library to plot and display the scores obtained
during training of a reinforcement learning agent. The function takes two arguments: scores, which is a list of
scores obtained in each game or episode, and mean_scores, which is a list of the mean scores obtained over a 
moving window of episodes.

The function starts by clearing the previous plot using display.clear_output(wait=True) and then displaying the
current figure using display.display(plt.gcf()). It then clears the current figure using plt.clf() and sets the
title, xlabel, and ylabel for the plot. The scores and mean_scores lists are then plotted using plt.plot().
The y-axis limits are set to start at 0 using plt.ylim(ymin=0).

Finally, the function adds text to the plot to display the current score and mean score at the end of the plot.
The plot is then shown using plt.show(block=False) and paused for a short amount of time using plt.pause(.1)
to allow for the plot to be updated.

Overall, this function is useful for monitoring the performance of a reinforcement learning agent during
training by plotting and displaying the scores obtained over time.
"""


import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


