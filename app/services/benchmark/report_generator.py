import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


def generate_benchmark_report(samples, metrics, output_path):

    df = pd.DataFrame(samples)

    # Create visualization
    plt.figure()
    plt.scatter(df["True X"], df["True Y"], label="True")
    plt.scatter(df["Predicted X"], df["Predicted Y"], label="Predicted")

    plt.legend()
    plt.title("Gaze Prediction Accuracy")

    plot_path = "benchmark_plot.png"
    plt.savefig(plot_path)

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(output_path)
    elements = []

    elements.append(Paragraph("Eye Tracking Benchmark Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            f"Mean Accuracy Error (px): {metrics['mean_accuracy_error_px']}",
            styles["BodyText"],
        )
    )

    elements.append(Spacer(1, 20))
    elements.append(Image(plot_path))

    doc.build(elements)

    return output_path