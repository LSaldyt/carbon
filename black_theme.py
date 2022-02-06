import plotly.graph_objects as go
import plotly.io as pio

plotly_template = pio.templates["plotly_dark"]

pio.templates["plotly_dark_custom"] = pio.templates["plotly_dark"]

pio.templates["plotly_dark_custom"].update({
    'layout' : {
#e.g. you want to change the background to transparent
    'paper_bgcolor': 'rgba(0,0,0,1.0)',
    'plot_bgcolor': 'rgba(0,0,0,1.0)'
    # 'paper_bgcolor': 'rgba(0,0,0,0.0)',
    # 'plot_bgcolor': 'rgba(0,0,0,0.0)'
    }})

pio.templates.default = "plotly_dark_custom"
