# 人脸识别模块
import face_recognition
# opencv-python
import cv2
import numpy as np
# 导入BOS相关模块
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient
# mqtt通信
import paho.mqtt.client as mqtt
import json
# 多线程
import threading
from time import sleep
# 树莓派模块
import RPi.GPIO as GPIO
import sys
# 功能说明：
# 1.先进行人脸识别，识别到陌生人，拍照上传云端
# 2.再接收手机端指令，接收到yes指令自动开门

# recv_name 作为全局变量，需要接收mqtt传入的值,传进rec(),Recv_name是局部变量
recv_name = "unknown"


# 发送图片到云端
class Bd_Storage(object):
    def __init__(self):
        # 设置BosClient的Host，Access Key ID和Secret Access Key
        self.bos_host = "bj.bcebos.com"  # 地址可以改，参考百度的python SDK文档
        self.access_key_id = "f882a84d6f2b4067bb071024347dbc17"
        self.secret_access_key = "2d9fb65a6bbc4a7bac7828850e3ca2ea"
        self.back_name = "bbbucket"

    def up_image(self, key_name, file):
        config = BceClientConfiguration(credentials=
                                        BceCredentials(self.access_key_id, self.secret_access_key),
                                        endpoint=self.bos_host)
        client = BosClient(config)

        key_name = key_name
        try:
            res = client.put_object_from_string(bucket=self.back_name, key=key_name, data=file)
        except Exception as e:
            return None
        else:
            print('put success!')


def send_photo(photo_address):
    with open(photo_address, 'rb') as f:
        bd = Bd_Storage()
        s = f.read()
        bd.up_image(photo_address, s)


def add_unknown_person(img_encoding):
    names['person' + str(i - 1)] = img_encoding  # 批量命名person 1 2 3…… 变量，将人脸编码赋值给变量
    exec('unknown_face_encodings.append(person{})'.format(i - 1))
    unknown_face_names.append('Unknow{}'.format(i - 1))




# 接受人名信息，添加熟人
def rec(Recv_name, img_encoding):
    # while True:
    # try:
    #     new_name = phone.recv(1024)
    # except BlockingIOError as e:
    #     new_name = None
    if Recv_name == "unknown":
        pass
    else:
        NEW_name = Recv_name.decode('utf-8')
        print(NEW_name)
        unknown_face_encodings.pop()
        unknown_face_names.pop()
        known_face_encodings.append(img_encoding)
        known_face_names.append(NEW_name)


def setServoAngle(servo, angle):
	pwm = GPIO.PWM(servo, 50)
	pwm.start(8)# 设置占空比
	dutyCycle = angle / 18. + 3.
	pwm.ChangeDutyCycle(dutyCycle) # 设置更新频率
	sleep(0.3)
	pwm.stop()

def servo():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    servo = 22 # int(sys.argv[1])
    GPIO.setup(servo, GPIO.OUT)
    setServoAngle(servo, 0)
    setServoAngle(servo, 90)
    GPIO.cleanup()

def servo02():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    servo = 27 # int(sys.argv[1])
    GPIO.setup(servo, GPIO.OUT)
    setServoAngle(servo, 0)
    setServoAngle(servo, 90)
    GPIO.cleanup()

def openDoor():
    t1 = threading.Thread(target=servo, args=())
    t2 = threading.Thread(target=servo02, args=())

    # 开启新线程
    t1.start()
    t2.start()
    t1.join()
    t2.join()

def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))


# def on_message01(mqttc, obj, msg):
#     print("send:" + msg.topic + " " + str(msg.payload))
#     datas = json.loads(msg.payload.decode('utf-8'))
#     print(datas)

def on_message02(mqttc, obj, msg):
    print("recv:" + msg.topic + " " + str(msg.payload))
    datas = json.loads(msg.payload.decode('utf-8'))
    print(datas)
    if datas == "yes": # 识别通过，开门
        openDoor()
    # # 接收人名，添加熟人
    # global recv_name
    # recv_name = datas


def on_publish(mqttc, obj, mid):
    print("publish:" + "mid: " + str(mid))


def on_subscribe(mqttc, obj, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))


def on_log(mqttc, obj, level, string):
    print(string)


def on_disconnect(mqttc, obj, rc):
    print("unsuccess connect %s" % rc)


# def send_word_device01():
#     HOST = 'ahekjhz.iot.gz.baidubce.com'
#     PORT = 1883
#     client_id = 'client_for_test'
#     username = 'thingidp@ahekjhz|device1|0|MD5'
#     password = '899e2ee82299559b2d6f04497cfc42fb'
#     topic = '$iot/device1/user/fortest'  # 能接受和发送
#     mqttc = mqtt.Client(client_id)
#     mqttc.username_pw_set(username, password)
#     mqttc.on_connect = on_connect
#     mqttc.on_message = on_message01 # 调用的仍然是on_message02,会和device02发生争抢
#     mqttc.on_publish = on_publish
#     # mqttc.on_subscribe = on_subscribe
#     mqttc.on_disconnect = on_disconnect
#     mqttc.connect(HOST, PORT, 60)
#
#     payload = '{}.jpg'.format(i)
#     payload = json.dumps(payload)
#     mqttc.publish(topic, payload, 0)
#     print("send message")
#     # bmqttc.subscribe(topic, 0)
#     #mqttc.loop_forever()


def recv_word_device02():
    HOST = 'ahekjhz.iot.gz.baidubce.com'
    PORT = 1883
    client_id = 'server-test'
    username = 'thingidp@ahekjhz|server-test|0|MD5'
    password = 'c556f3ba7effc76db8225cfcb3e2cad1'
    topic = '$iot/server-test/user/fortest'  # 能接受和发送
    mqttc02 = mqtt.Client(client_id)
    mqttc02.username_pw_set(username, password)
    mqttc02.on_connect = on_connect
    mqttc02.on_message = on_message02
    # mqttc02.on_publish = on_publish
    mqttc02.on_subscribe = on_subscribe
    mqttc02.on_disconnect = on_disconnect
    mqttc02.connect(HOST, PORT, 600)
    mqttc02.subscribe(topic, 0)
    mqttc02.loop_forever()


# 人脸识别部分
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("02.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# nikesong unknown test person
nike_image = face_recognition.load_image_file("03.jpg")
nike_face_encoding = face_recognition.face_encodings(nike_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "zhang"
]

unknown_face_encodings = [
    nike_face_encoding
]

unknown_face_names = [
    "Nike"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
#
i = 0
j = 0

names = globals()

temp_face_encoding = globals()
known_person_coming = False
last_person_name = "zhang"


t2 = threading.Thread(target=recv_word_device02, args=())


# 开启新线程
t2.start()  # 开始监听信息


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        j = i
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            unknown_matches = face_recognition.compare_faces(unknown_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            unknown_face_distances = face_recognition.face_distance(unknown_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            unknown_best_match_index = np.argmin(unknown_face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            elif unknown_matches[unknown_best_match_index]:
                name = unknown_face_names[unknown_best_match_index]

            else:
                # photo_address = 'E:/_TempPhoto/' + str(i) + '.jpg'
                photo_address = '/home/pi/code/pic/' + str(i) + '.jpg'

                cv2.imwrite(photo_address, frame)  # 识别到unkown时拍照
                temp_face_encoding = face_encoding
                send_photo(photo_address)  # 上传图片到云端
                i += 1
            # 传回未知人姓名，添加
                global temp_face_encoding
                rec(recv_name, temp_face_encoding)

            # trd = threading.Thread(target=rec, args=(phone, face_encoding))
            # trd.start()
            face_names.append(name)

    process_this_frame = not process_this_frame
    if i > j:
        add_unknown_person(face_encoding)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


t2.join()  # 监听结束

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
