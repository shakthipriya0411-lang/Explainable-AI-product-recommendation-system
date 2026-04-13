import os


def save_uploaded_image(file):

    if not file:
        return None

    folder = "static/uploads"

    if not os.path.exists(folder):
        os.makedirs(folder)

    filepath = os.path.join(
        folder,
        file.filename
    )

    file.save(filepath)

    return filepath