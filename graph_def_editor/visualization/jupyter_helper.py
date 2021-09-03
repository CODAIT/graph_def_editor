"""Helper methods to display gde graph in Jupyter Notebook or Colab."""

import time
from IPython.display import HTML


def jupyter_show_as_svg(dg):
  """Shows object as SVG (by default it is rendered as image).

  Args:
    dg: digraph object

  Returns:
    Graph rendered in SVG format
  """
  return HTML(dg.pipe(format="svg").decode("utf-8"))


def jupyter_pan_and_zoom(
    dg,
    element_styles="height:auto",
    container_styles="overflow:hidden",
    pan_zoom_json="{controlIconsEnabled: true, zoomScaleSensitivity: 0.4, "
    "minZoom: 0.2}"):
  """Embeds SVG object into Jupyter cell with ability to pan and zoom.

  Args:
    dg: digraph object
    element_styles: CSS styles for embedded SVG element.
    container_styles: CSS styles for container div element.
    pan_zoom_json: pan and zoom settings, see
      https://github.com/bumbu/svg-pan-zoom

  Returns:
    Graph rendered as HTML using javascript for Pan and Zoom functionality.
  """
  svg_txt = dg.pipe(format="svg").decode("utf-8")
  html_container_class_name = F"svg_container_{int(time.time())}"
  html = F"""
        <div class="{html_container_class_name}">
            <style>
                .{html_container_class_name} {{
                    {container_styles}
                }}
                .{html_container_class_name} SVG {{
                    {element_styles}
                }}
            </style>
            <script src="https://bumbu.me/svg-pan-zoom/dist/svg-pan-zoom.min.js"></script>
            <script type="text/javascript">
                attempts = 5;
                var existCondition = setInterval(function() {{
                  console.log(attempts);
                  svg_el = document.querySelector(".{html_container_class_name} svg");
                  if (svg_el != null) {{
                      console.log("Exists!");
                      clearInterval(existCondition);
                      svgPanZoom(svg_el, {pan_zoom_json});
                  }}
                  if (--attempts == 0) {{
                      console.warn("SVG element not found, zoom wont work");
                      clearInterval(existCondition);
                  }}
                }}, 100); // check every 100ms
            </script>
            {svg_txt}
        </div>
    """
  return HTML(html)
