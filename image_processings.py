import os
import cv2

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def divide_image(image_path):
    try:
        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Define the bounding box coordinates of the table
        bounding_box = (274, 1565, 2293, 2503)

        # Define the row coordinates
        num_rows = 6
        starting_height = 130
        reduction_factor = 0.8

        # Calculate the reduced row height
        table_height = bounding_box[3] - bounding_box[1]
        reduced_row_height = int((table_height - starting_height) / num_rows * reduction_factor)

        # Create the root directory for rows
        root_dir = "row"
        create_directory_if_not_exists(root_dir)

        for i in range(num_rows):
            row_start = bounding_box[1] + starting_height + i * reduced_row_height
            row_end = bounding_box[1] + starting_height + (i + 1) * reduced_row_height
            row_img = image[row_start:row_end, bounding_box[0]:bounding_box[2]]

            # Create the directory for the current row
            row_dir = os.path.join(root_dir, f"row_{i+1}")
            create_directory_if_not_exists(row_dir)

            if i == 0:
                # For the first row, save it as a file instead of a folder
                row_img_path = f"{row_dir}.png"
                cv2.imwrite(row_img_path, row_img)
            else:
                # Divide the row image into half
                half_width = row_img.shape[1] // 2
                half_row_img1 = row_img[:, :half_width]
                half_row_img2 = row_img[:, half_width:]

                # Save the first half of the row as separate cell images
                half1_dir = os.path.join(row_dir, "half1")
                create_directory_if_not_exists(half1_dir)
                for j in range(5):
                    cell_start = j * (half_width // 5)
                    cell_end = (j + 1) * (half_width // 5)
                    cell_img = half_row_img1[:, cell_start:cell_end]
                    cell_img_path = os.path.join(half1_dir, f"cell_{j+1}.png")
                    cv2.imwrite(cell_img_path, cell_img)

                # Save the second half of the row as separate cell images
                half2_dir = os.path.join(row_dir, "half2")
                create_directory_if_not_exists(half2_dir)
                for j in range(5):
                    cell_start = j * (half_width // 5)
                    cell_end = (j + 1) * (half_width // 5)
                    cell_img = half_row_img2[:, cell_start:cell_end]
                    cell_img_path = os.path.join(half2_dir, f"cell_{j+1}.png")
                    cv2.imwrite(cell_img_path, cell_img)

    except Exception as e:
        print(f"Error: {e}")
