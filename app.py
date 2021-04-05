import cv2 as cv
import rospy
import numpy as np
import roslib
roslib.load_manifest('ui_interpretation')
import tensorflow as tf
from cv_bridge import CvBridge, CvBridgeError
from ui_interpretation.msg import localizationMsg,uiOutMsg
from sensor_msgs.msg import Image
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1

# Global Variables----------------------------------------------------------
bridge = CvBridge()
rgbImg = np.zeros([640,480],dtype = np.uint8)
grayImg = np.zeros([640,480], dtype = np.uint8)
threshImg = np.zeros([640,480], dtype = np.uint8)
g_input_height = 96
g_input_width = 96
g_input_mean = 0
g_input_std = 255


rHour1 = []
rHour2 = []
rHour3 = []
rCentr1 = []
rCentr2 = []
rCentr3 = []
rCentr4 = []
rKg1 = []
rKg2 = []
rTemp1 = []
rTemp2 = []
rSpeed = []
rSymbol1 = []
rSymbol2 = []
rSymbol3 =[]
rSymbol4 =[]
rSymbol5 = []
rSymbol6 = []
rSymbol7 = []
rSymbol8 = []
rSymbol9 = []
rSymbol10 = []
rSymbol11 = []
rSymbol12 = []
rSymbol13 = []
rProgram1 = []
rProgram2 = []
rProgram3 = []
rProgram4 = []
rProgram5 = []
rProgram6 = []
rProgram7 = []
rProgram8 = []
rProgram9 = []
rProgram10 = []
#---------------------------------------------------------------------------


class SceneCoordinates:
    def __init__(self):
        print("Coordinates are being fetched now")

    def sceneCoordCallback(self,data):
        # This method used for setting ROIs on the scene image.
        global rHour1,rHour2,rHour3
        global rCentr1,rCentr2,rCentr3,rCentr4
        global rKg1,rKg2
        global rTemp1,rTemp2
        global rSpeed
        global rSymbol1,rSymbol2,rSymbol3,rSymbol4,rSymbol5,rSymbol6,rSymbol7,rSymbol8,rSymbol9,rSymbol10
        global rSymbol11,rSymbol12,rSymbol13
        global rProgram1,rProgram2,rProgram3,rProgram4,rProgram5,rProgram6,rProgram7,rProgram8,rProgram9,rProgram10
        # Hour
        rHour1 = [int(data.firsthourupleft[0]), int(data.firsthourupleft[1]), int(data.firsthourbottomleft[1]), int(data.firsthourupright[0])]
        rHour2 = [int(data.secondhourupleft[0]), int(data.secondhourupleft[1]), int(data.secondhourbottomleft[1]), int(data.secondhourupright[0])]
        rHour3 = [int(data.thirdhourupleft[0]), int(data.thirdhourupleft[1]), int(data.thirdhourbottomleft[1]), int(data.thirdhourupright[0])]
        # Centrifugation
        rCentr1 = [int(data.firstcentrupleft[0]), int(data.firstcentrupleft[1]), int(data.firstcentrbottomleft[1]), int(data.firstcentrupright[0])]
        rCentr2 = [int(data.secondcentrupleft[0]), int(data.secondcentrupleft[1]), int(data.secondcentrbottomleft[1]), int(data.secondcentrupright[0])]
        rCentr3 = [int(data.thirdcentrupleft[0]), int(data.thirdcentrupleft[1]), int(data.thirdcentrbottomleft[1]), int(data.thirdcentrupright[0])]
        rCentr4 = [int(data.fourthcentrupleft[0]), int(data.fourthcentrupleft[1]), int(data.fourthcentrbottomleft[1]), int(data.fourthcentrupright[0])]
        # Kilogram
        rKg1 = [int(data.firstkgupleft[0]), int(data.firstkgupleft[1]), int(data.firstkgbottomleft[1]), int(data.firstkgupright[0])]
        rKg2 = [int(data.secondkgupleft[0]), int(data.secondkgupleft[1]), int(data.secondkgbottomleft[1]), int(data.secondkgupright[0])]
        # Temperature
        rTemp1 = [int(data.firsttempupleft[0]), int(data.firsttempupleft[1]), int(data.firsttempbottomleft[1]), int(data.firsttempupright[0])]
        rTemp2 = [int(data.secondtempupleft[0]), int(data.secondtempupleft[1]), int(data.secondtempbottomleft[1]), int(data.secondtempupright[0])]
        # Speed
        rSpeed = [int(data.speedupleft[0]), int(data.speedupleft[1]), int(data.speedbottomleft[1]), int(data.speedupright[0])]
        # Symbols
        rSymbol1 = [int(data.s1upleft[0]), int(data.s1upleft[1]), int(data.s1bottomleft[1]), int(data.s1upright[0])]
        rSymbol2 = [int(data.s2upleft[0]), int(data.s2upleft[1]), int(data.s2bottomleft[1]), int(data.s2upright[0])]
        rSymbol3 = [int(data.s3upleft[0]), int(data.s3upleft[1]), int(data.s3bottomleft[1]), int(data.s3upright[0])]
        rSymbol4 = [int(data.s4upleft[0]), int(data.s4upleft[1]), int(data.s4bottomleft[1]), int(data.s4upright[0])]
        rSymbol5 = [int(data.s5upleft[0]), int(data.s5upleft[1]), int(data.s5bottomleft[1]), int(data.s5upright[0])]
        rSymbol6 = [int(data.s6upleft[0]), int(data.s6upleft[1]), int(data.s6bottomleft[1]), int(data.s6upright[0])]
        rSymbol7 = [int(data.s7upleft[0]), int(data.s7upleft[1]), int(data.s7bottomleft[1]), int(data.s7upright[0])]
        rSymbol8 = [int(data.s8upleft[0]), int(data.s8upleft[1]), int(data.s8bottomleft[1]), int(data.s8upright[0])]
        rSymbol9 = [int(data.s9upleft[0]), int(data.s9upleft[1]), int(data.s9bottomleft[1]), int(data.s9upright[0])]
        rSymbol10 = [int(data.s10upleft[0]), int(data.s10upleft[1]), int(data.s10bottomleft[1]), int(data.s10upright[0])]
        rSymbol11 = [int(data.s11upleft[0]), int(data.s11upleft[1]), int(data.s11bottomleft[1]), int(data.s11upright[0])]
        rSymbol12 = [int(data.s12upleft[0]), int(data.s12upleft[1]), int(data.s12bottomleft[1]), int(data.s12upright[0])]
        rSymbol13 = [int(data.s13upleft[0]), int(data.s13upleft[1]), int(data.s13bottomleft[1]), int(data.s13upright[0])]
        # Program Recognition
        rProgram1 = [int(data.prg1left[0]),int(data.prg1left[1]),int(data.prg2left[1]),int(data.prg1right[0])]
        rProgram2 = [int(data.prg2left[0]),int(data.prg2left[1]),int(data.prg3left[1]),int(data.prg2right[0])]
        rProgram3 = [int(data.prg3left[0]),int(data.prg3left[1]),int(data.prg4left[1]),int(data.prg3right[0])]
        rProgram4 = [int(data.prg4left[0]),int(data.prg4left[1]),int(data.prg5left[1]),int(data.prg4right[0])]
        rProgram5 = [int(data.prg5left[0]),int(data.prg5left[1]),int(data.prg6left[1]),int(data.prg5right[0])]
        rProgram6 = [int(data.prg6left[0]),int(data.prg6left[1]),int(data.prg7left[1]),int(data.prg6right[0])]
        rProgram7 = [int(data.prg7left[0]),int(data.prg7left[1]),int(data.prg8left[1]),int(data.prg7right[0])]
        rProgram8 = [int(data.prg8left[0]),int(data.prg8left[1]),int(data.prg9left[1]),int(data.prg8right[0])]
        rProgram9 = [int(data.prg9left[0]),int(data.prg9left[1]),int(data.prg10left[1]),int(data.prg9right[0])]
        rProgram10 = [int(data.prg10left[0]),int(data.prg10left[1]),int(data.prg11left[1]),int(data.prg10right[0])]


    def checkStatus(self, rElement):
        # checking on/off status of program and symbols on the user interface
        status = False
        croppedROI = threshImg[rElement[1]:rElement[2], rElement[0]:rElement[3]]
        contours, hierarchy = cv.findContours(croppedROI,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv.contourArea, reverse=True)[:10]
        try:
            perimeter = cv.arcLength(contours[0], True)
            if perimeter>10:
                status = True
            else:
                status = False
        except Exception as exc:
            print(exc)

        return status

class RosMessageSetting:
    def __init__(self):
        print("Ros message started to be prepared")
    def setRosMessageForDigit(self,msg,interpretation_results):
        spin_number = interpretation_results[0]
        kg = interpretation_results[1]
        temperatureWashing = interpretation_results[2]
        time = interpretation_results[3]
        spinSpeed = interpretation_results[4]

        msg.User_select_Rinse_Nums = spin_number
        msg.Kilogram = kg
        msg.temperature = temperatureWashing
        msg.User_Select_Delay_Time = time
        msg.Spin_speed = spinSpeed

    def setRosMessageForProgram(self,msg,scene_coords):
        #checking whether program led is active or inactive
        if scene_coords.checkStatus(rProgram1) == True:
            msg.program = 1
        elif scene_coords.checkStatus(rProgram2) == True:
            msg.program = 2
        elif scene_coords.checkStatus(rProgram3) == True:
            msg.program = 3
        elif scene_coords.checkStatus(rProgram4) == True:
            msg.program = 4
        elif scene_coords.checkStatus(rProgram5) == True:
            msg.program = 5
        elif scene_coords.checkStatus(rProgram6) == True:
            msg.program = 6
        elif scene_coords.checkStatus(rProgram7) == True:
            msg.program = 7
        elif scene_coords.checkStatus(rProgram8) == True:
            msg.program = 8
        elif scene_coords.checkStatus(rProgram9) == True:
            msg.program = 9
        elif scene_coords.checkStatus(rProgram10) == True:
            msg.program = 10
        else:
            print("Error occured or the washing machine is not active!!!")

    def setRosMessageForOption(self,msg,scene_coords):
        if scene_coords.checkStatus(rOption1) == True:
            msg.option = 1
        elif scene_coords.checkStatus(rOption2) == True:
            msg.option = 2
        elif scene_coords.checkStatus(rOption3) == True:
            msg.option = 3
        else:
            print("Error occured or the washing machine is not active!!!")

    def setRosMessageForSymbols(self,msg,scene_coords):
        isS1_OnOff = scene_coords.checkStatus(rSymbol1)
        isS2_OnOff = scene_coords.checkStatus(rSymbol2)
        isS3_OnOff = scene_coords.checkStatus(rSymbol3)
        isS4_OnOff = scene_coords.checkStatus(rSymbol4)
        isS5_OnOff = scene_coords.checkStatus(rSymbol5)
        isS6_OnOff = scene_coords.checkStatus(rSymbol6)
        isS7_OnOff = scene_coords.checkStatus(rSymbol7)
        isS8_OnOff = scene_coords.checkStatus(rSymbol8)
        isS9_OnOff = scene_coords.checkStatus(rSymbol9)
        isS10_OnOff = scene_coords.checkStatus(rSymbol10)
        isS11_OnOff = scene_coords.checkStatus(rSymbol11)
        #NormalTM2L7 or DailyTM2L7
        isS12_OnOff = scene_coords.checkStatus(rSymbol12)
        isS13_OnOff = scene_coords.checkStatus(rSymbol13)

        msg.Washing = isS1_OnOff
        msg.Rinsing = isS2_OnOff
        msg.Spinning = isS3_OnOff
        msg.Steaming = isS5_OnOff
        msg.Steam_Anticrease = isS6_OnOff
        msg.ExtraRinse = isS7_OnOff
        msg.Night_Cycle = isS8_OnOff
        msg.Prewash = isS9_OnOff
        msg.Stain = isS10_OnOff
        msg.Economy = isS11_OnOff
        if isS12_OnOff == False:
            msg.NormalTM2L7 = True
            msg.DailyTM2L7 = False
            msg.SuperQuickTM2L7 = False
        else:
            msg.NormalTM2L7 = False
            msg.DailyTM2L7 = True
        msg.Low_Steam = False
        msg.Medium_Steam = False
        msg.Delay = True

def preClassification(scene_coords, p_label):
    # Centrifugation
    centrifugation = {}
    kilogram = {}
    temperature = {}
    hour = {}
    speed = {}

    if scene_coords.checkStatus(rCentr1)==True:
        centrifugation[0] = classification(rCentr1, p_label)
    else:
        centrifugation[0] = 0
    if scene_coords.checkStatus(rCentr2)==True:
        centrifugation[1] = classification(rCentr2, p_label)
    else:
        centrifugation[1] = 0
    if scene_coords.checkStatus(rCentr3)==True:
        centrifugation[2] = classification(rCentr3, p_label)
    else:
        centrifugation[2] = 0
    if scene_coords.checkStatus(rCentr4)==True:
        centrifugation[3] = classification(rCentr4, p_label)
    else:
        centrifugation[3] = 0
    # Kilogram
    if scene_coords.checkStatus(rKg1)==True:
        kilogram[0] = classification(rKg1, p_label)
    else:
        kilogram[0] = 0
    if scene_coords.checkStatus(rKg2)==True:
        kilogram[1] = classification(rKg2, p_label)
    else:
        kilogram[1] = 0
    # Temperature
    if scene_coords.checkStatus(rTemp1)==True:
        temperature[0] = classification(rTemp1, p_label)
    else:
        temperature[0] = 0
    if scene_coords.checkStatus(rTemp1)==True:
        temperature[1] = classification(rTemp2, p_label)
    else:
        temperature[1] = 0
    # Hour
    if scene_coords.checkStatus(rHour1)==True:
        hour[0] = classification(rHour1, p_label)
    else:
        hour[0] = 0
    if scene_coords.checkStatus(rHour2)==True:
        hour[1] = classification(rHour2, p_label)
    else:
        hour[1] = 0
    if scene_coords.checkStatus(rHour3)==True:
        hour[2] = classification(rHour3, p_label)
    else:
        hour[2] = 0
    # Speed
    if scene_coords.checkStatus(rSpeed)==True:
        speed[0] = classification(rSpeed, p_label)
    else:
        speed[0] = '0'

    spin_number = int(str(centrifugation[0])+str(centrifugation[1])+str(centrifugation[2])+str(centrifugation[3]))
    kg = float(str(kilogram[0])+'.'+str(kilogram[1]))
    temperatureWashing = int(str(temperature[0])+str(temperature[1]))
    time = int(str(hour[0]) + str(hour[1]) + str(hour[2]))
    spinSpeed = int(speed[0])
    interpretation_results = [spin_number,kg,temperatureWashing,time,spinSpeed]
    return interpretation_results

def frameCallback(data):
    #Subscribing Video Frames
    global rgbImg
    global grayImg
    global threshImg
    try:
        rgbImg = bridge.imgmsg_to_cv2(data,"bgr8")
        grayImg = cv.cvtColor(rgbImg, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(grayImg,(3,3),0)
        #Otsu's thresholding after Gaussian filtering
        ret3,threshImg = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imshow("mfdsd", threshImg)
        cv.waitKey(3)
    except CvBridgeError as e:
        print(e)

def load_labels(label_file):
    #loading labels
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def read_tensor_from_image_file(
        image,
        input_height=g_input_height,
        input_width=g_input_width,
        input_mean=g_input_mean,
        input_std=g_input_std):
    # Numpy array
    np_image_data = np.asarray(image)
    np_image_data = np.stack((np_image_data,)*3, -1)
    float_caster = tf.cast(np_image_data, tf.float32)
    # maybe insert float convertion here - see edit remark!
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    model = tf.keras.model.load_model("desired_digitLENETSlowFeatureExtraction/saved_model.pb")
    result = model(normalized)
    return result

def classification(rElement,label):
    croppedROI = threshImg[rElement[1]:rElement[2], rElement[0]:rElement[3]]
    t = read_tensor_from_image_file(
        croppedROI,
        input_height=g_input_height,
        input_width=g_input_width,
        input_mean=g_input_mean,
        input_std=g_input_std)

    print(t)
    
    results = np.squeeze(t)
    reply = '0'
    top_k = results.argsort()[-5:][::-1]
    if (str(labels[top_k[0]]) == '0' or str(labels[top_k[0]]) == '-'):
        reply = '0'
    elif (str(labels[top_k[0]]) == '1'):
        reply = '1'
    elif (str(labels[top_k[0]]) == '2'):
        reply = '2'
    elif (str(labels[top_k[0]]) == '3'):
        reply = '3'
    elif (str(labels[top_k[0]]) == '4'):
        reply = '4'
    elif (str(labels[top_k[0]]) == '5'):
        reply = '5'
    elif (str(labels[top_k[0]]) == '6'):
        reply = '6'
    elif (str(labels[top_k[0]]) == '7'):
        reply = '7'
    elif (str(labels[top_k[0]]) == '8'):
        reply = '8'
    elif (str(labels[top_k[0]]) == '9'):
        reply = '9'
    return reply




if __name__ == "__main__":

    # Path for neural network to load
    
    model_file =  '/home/dlar/catkin_ws/src/ui_interpretation/Data/graph/saved_model.pb'
    label_file =  '/home/dlar/catkin_ws/src/ui_interpretation/Data/graph/labels.txt'
    # Initialize ROS
    rospy.init_node("digit_interpretation", anonymous=True)
    rate = rospy.Rate(5)
    scene_coord = SceneCoordinates()
    uiScreenMsg = uiOutMsg()
    # Topic for subscribing images
    imageTopic = "/camera/rgb/image_raw"
    # Topic for subscribing scene coordinates
    coordSubTopic = "/scene_coords"
    # ROS Subscriber and Publisher
    image_sub = rospy.Subscriber(imageTopic, Image, frameCallback, queue_size=1)
    roi_sub = rospy.Subscriber(coordSubTopic,localizationMsg, scene_coord.sceneCoordCallback, queue_size=1)
    ui_pub = rospy.Publisher("ui_estimation", uiOutMsg, queue_size=1)
    # Graph file and labels are loaded
    #tf.compat.v1.disable_eager_execution()    
    labels = load_labels(label_file)
    while not rospy.is_shutdown():
        #interpretation_results = preClassification(scene_coord,labels)
        #RosMessageSetting().setRosMessageForDigit(uiScreenMsg,interpretation_results)
        #RosMessageSetting().setRosMessageForSymbols(uiScreenMsg,scene_coord)
        #RosMessageSetting().setRosMessageForProgram(uiScreenMsg,scene_coord)
        #  RosMessageSetting().setRosMessageForOption(uiScreenMsg,scene_coord)
       
        
        ui_pub.publish(uiScreenMsg)
        rate.sleep()
    cv.release()
    cv.destroyAllWindows()