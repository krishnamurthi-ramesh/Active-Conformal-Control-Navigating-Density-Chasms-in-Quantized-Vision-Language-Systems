from PIL import Image
img = Image.open('/home/cse-sdpl/research/Active-Conformal-Control-Navigating-Density-Chasms-in-Quantized-Vision-Language-Systems/TMLCN_LaTeX_Template/qualitative_analysis.png')
# Crop: (left, upper, right, lower)
cropped = img.crop((0, 0, 1024, 420))
cropped.save('/home/cse-sdpl/research/Active-Conformal-Control-Navigating-Density-Chasms-in-Quantized-Vision-Language-Systems/TMLCN_LaTeX_Template/qualitative_chart_cropped.png')
