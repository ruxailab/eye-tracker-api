# To be added later on the project
import numpy as np

def generate_heatmap(predictions, width, height, bins=(64, 64), sigma=1.5):
	"""
	Generate a simple 2D density heatmap from predictions.

	Args:
		predictions (list): list of dicts with keys `x` and `y` (pixels).
		width (int): screen width in pixels.
		height (int): screen height in pixels.
		bins (tuple): bins for (x_bins, y_bins).
		sigma (float): gaussian smoothing sigma in bins.

	Returns:
		list(list(float)): 2D array (y major) normalized to 0..1 suitable for JSON.
	"""
	if not predictions or not width or not height:
		return None

	xs = []
	ys = []
	for p in predictions:
		x = p.get("x") or p.get("predicted_x")
		y = p.get("y") or p.get("predicted_y")
		if x is None or y is None:
			continue
		xs.append(x)
		ys.append(y)

	if len(xs) == 0:
		return None

	x_bins, y_bins = bins
	# Use numpy histogram2d (note: histogram2d expects x, y)
	heat, xedges, yedges = np.histogram2d(xs, ys, bins=[x_bins, y_bins], range=[[0, width], [0, height]])

	# transpose so rows correspond to y (top->bottom)
	heat = heat.T

	# gaussian smoothing in frequency domain (approx)
	try:
		from scipy.ndimage import gaussian_filter

		heat = gaussian_filter(heat, sigma=sigma)
	except Exception:
		# fallback: simple local normalization if scipy not available
		pass

	# normalize
	mn = float(np.min(heat))
	mx = float(np.max(heat))
	if mx - mn > 0:
		heat = (heat - mn) / (mx - mn)
	else:
		heat = heat * 0.0

	return heat.tolist()


def attach_heatmap_to_payload(payload, bins=(64, 64), sigma=1.5):
	"""Attach generated heatmap to the payload produced by `predict_new_data_simple`.

	Returns a modified payload (copy) with `heatmap` populated if possible.
	"""
	if not payload or "predictions" not in payload:
		return payload

	screen = payload.get("screen") or {}
	width = screen.get("width")
	height = screen.get("height")

	heat = generate_heatmap(payload["predictions"], width, height, bins=bins, sigma=sigma)
	out = dict(payload)
	out["heatmap"] = heat
	return out
# To be added later on the project