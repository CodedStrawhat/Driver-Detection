# Driver-Detection

Road accidents take place all over the world with over 5 crore people getting injuries due to them every year. Over 10 lakh people suffer from fatal road accidents every year around the Globe. Most of these accidents take place in low to middle-income countries like India and cost them about 3% of their GDP. Around 20% of traffic accidents are attributed to distracted or drowsy drivers. People fear hiring drivers they don't know to drive for them, in the fear of such accidents. Tackling these issues can be difficult when our lifestyle does not align with avoiding drowsy driving. There is no efficient method deployed till now to alert drowsy or distracted drivers in real-time to prevent accidents. 

**We have proposed a solution for this problem**

In-vehicle driver monitoring systems are going to be a crucial element of vehicles of the future. One important function of such systems is to determine if the driver is paying attention to the road. If not, the system alerts the driver to bring his/her attention back to the road. Some ways to accomplish this are to monitor the driver's eyes and the gaze of their eyes, also to check for single-handed driving. Our solution to this would be based on Computer Vision and Machine Learning to make a device to check for drowsy or distracted driving and alert the driver to bring their attention back to the road.

Weights for the model : https://drive.google.com/file/d/1ZbAz4pGvrnVrh-rhxLkAgTMvHKjlCbAO/view?usp=sharing

**File descriptions**

beep.wav - The sound that will be played to alert the driver(can be replaced by any other)
hack_tag_sideview_modeltrainer.ipynb - Notebook file for Colab notebook used for training the model
hacktag.py - Python Program to detect face and check for drowsiness
hacktag_SideCam.py - Python program to use the ML model and classify video feed as safe driving or distracted driving
hacktag_sideview_modeltrain.py - .Py file for colab notebook used for training the model
requirements.txt - For hacktag.py
requirements_side.txt - For hacktag_SideCam.py
