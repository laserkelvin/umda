
import networkx as nx
from bokeh.io import output_file, show
from bokeh.models import Ellipse, Circle, StaticLayoutProvider, MultiLine, HoverTool, BoxZoomTool, ResetTool, EdgesAndLinkedNodes, BoxSelectTool, TapTool, NodesAndLinkedEdges
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, from_networkx


def bokehify_graph(nx_graph, title: str, path: str, show_plot: bool = True):
    plot = figure(title=title, x_range=(-1.1,1.1), y_range=(-1.1,1.1), plot_width=1000, plot_height=700)

    node_hover_tool = HoverTool(tooltips=[("A", "@A"), ("B", "@B"), ("Log10 rate", "@rate")])
    plot.add_tools(node_hover_tool, TapTool(), BoxZoomTool(), ResetTool(), BoxSelectTool())

    graph_renderer = from_networkx(nx_graph, nx.spring_layout, scale=2, center=(0,0), seed=42)

    graph_renderer.node_renderer.glyph = Circle(size=5, fill_color=Spectral4[0])
    graph_renderer.node_renderer.selection_glyph = Circle(size=15., fill_color=Spectral4[2])
    graph_renderer.edge_renderer.glyph = MultiLine(line_color="react_color", line_alpha=0.3, line_width=0.5)
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color='#abdda4', line_width=3, line_alpha=0.6)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=3, line_alpha=0.8)

    graph_renderer.inspection_policy = EdgesAndLinkedNodes()
    graph_renderer.selection_policy = NodesAndLinkedEdges()
    plot.renderers.append(graph_renderer)

    output_file(f"{path}.html")
    if show_plot:
        show(plot)
    return plot