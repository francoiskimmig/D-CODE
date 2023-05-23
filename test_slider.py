import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Create some data to plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot and axis objects
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

# Plot the initial data
line, = ax.plot(x, y, label="sin(x)")

# Set the initial slider values
slider1_init_val = 0.0
slider2_init_val = 0.0

# Define the update function as a closure that takes non-local variables
def update_func(slider1_val, slider2_val, line):
    def update(val):
        nonlocal slider1_val, slider2_val
        # Update the slider values
        slider1_val = slider1.val
        slider2_val = slider2.val
        # Update the y-data based on the new slider values
        y_new = np.sin(x + slider1_val) + slider2_val
        # Update the plot data
        line.set_data(x, y_new)
        # Set the new y-axis limits
        ax.set_ylim([y_new.min(), y_new.max()])
        # Redraw the plot
        fig.canvas.draw_idle()
    return update

# Create the sliders
ax_slider1 = plt.axes([0.2, 0.2, 0.6, 0.03])
slider1 = Slider(ax_slider1, 'Slider 1', 0, 10, valinit=slider1_init_val)

ax_slider2 = plt.axes([0.2, 0.15, 0.6, 0.03])
slider2 = Slider(ax_slider2, 'Slider 2', -1, 1, valinit=slider2_init_val)

# Define the update function as a closure and connect it to the sliders
update = update_func(slider1_init_val, slider2_init_val, line)
slider1.on_changed(update)
slider2.on_changed(update)

# Display the plot
plt.show()