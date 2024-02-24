# b4 running pip install Pillow
# open terminal python3 home\impactframes\SD-forge\extensions\IF_prompt_MKR\uuid_image_renamer.py change to your path
import os
import uuid
from tkinter import Tk, filedialog, StringVar, Button, messagebox
from PIL import Image

def rename_and_convert_images(folder_path):
    """Renames images with UUIDs and converts them to JPG."""
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".webp"):
            file_path = os.path.join(folder_path, filename)
            file_ext = os.path.splitext(filename)[1]

            # Rename
            new_filename = str(uuid.uuid4()) + ".jpg"  
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(file_path, new_file_path)

            # Convert (if not already JPG)
            if file_ext.lower() != ".jpg":
                try:
                    img = Image.open(new_file_path)
                    img.save(new_file_path, format="JPEG")
                except OSError:
                    pass  

    messagebox.showwarning("Conversion Notice", "Renaming completed. All images have been converted to JPG. .")

def select_folder():
    """Opens a folder selection dialog and updates the folder path."""
    root = Tk()
    root.withdraw()  
    folder_selected = filedialog.askdirectory()
    if folder_selected: 
        if messagebox.askyesno("Confirmation", "This script will convert all images in the selected folder to JPG. Do you want to continue?"):
            folder_path.set(folder_selected)
            rename_and_convert_images(folder_selected)
        else:
            messagebox.showinfo("Process Cancelled", "The process has been cancelled by the user.")


root = Tk()
root.title("Image Renamer")

folder_path = StringVar()  

select_button = Button(root, text="Select Folder", command=select_folder)
select_button.pack()

root.mainloop()