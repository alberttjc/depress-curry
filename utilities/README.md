# Utility Functions

The following scripts were used in the creation of the dataset for UTD_MHAD for Human Activity Recognition - 2D Pose Input.
They were run in the listed order below.
Please note that any directory references will need to be changed before use, and that no liability is taken. Please read the code before using it.

1. video2frame.py   :     Converts videos into frames to be processed
2. run\_openpose.py :     Runs openpose on all images into JSON format
3. json2text.py     :     Convert JSON files into text format
4. labler.py        :     Creates a list of labels for each action, in txt format
5. createDataset.py :     Splits dataset into training and testing set

(Optional)
frameRename.py      :     Rename frames into numerical order
