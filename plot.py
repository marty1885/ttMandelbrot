import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('benchmark.csv', index_col=0)

sns.set_style("darkgrid")
# plt.yscale('log')

plt.title('Time to render mandelbrot set')
sns.lineplot(data=df, x='size', y='time', hue='executable')
plt.show()
