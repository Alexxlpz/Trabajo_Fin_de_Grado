import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState, useRef } from 'react';
import { Modal, Button, StyleSheet, Text, TouchableOpacity, View, Alert, Image } from 'react-native';
import { Fontisto } from '@expo/vector-icons';
import Constants from "expo-constants";

export default function Camera() {
    const [recording, setRecording] = useState<boolean>(true);
    const [permission, requestPermission] = useCameraPermissions();
    const cameraRef = useRef<CameraView>(null);

    const [modalVisible, setModalVisible] = useState(false);
    const [resultData, setResultData] = useState<{number: number, imageb64: string} | null>(null); // guardamos el return que nos devuelve el backend, que contiene el numero de objetos detectados y la imagen en b64



    if (!permission) {
        // Camera permissions are still loading.
        return <View />;
    }

    if (!permission.granted) {
        // Camera permissions are not granted yet.
        return (
            <View style={styles.container}>
                <Text style={styles.message}>We need your permission to show the camera</Text>
                <Button onPress={requestPermission} title="grant permission" />
            </View>
        );
    }

    function toggleCameraRecording(){
        setRecording(current => (current === false));
    }

    async function takePicture(){
         if (cameraRef.current) {
             try {
                 const options = { quality: 0.7, base64: true };
                 const data = await cameraRef.current.takePictureAsync(options);
                 console.log('Imagen capturada:');

                 // si el dato no es null se lo pasamos a la función para que lo mande al backend y lo analice
                 //console.log(data);
                 if (data !== null) {
                     //console.log('antes de enviar la imagen');
                     await fetchPicture(data.base64);
                     console.log('Imagen enviada al servidor');
                 }

             } catch (error) {
                 console.error("Error al tomar la foto:", error);
             }
         }
    }

    async function fetchPicture(base64Data: string){
            try {
                // lo enviamos en una peticion post ya que es exageradamente larga la cadena de b64
              const response = await fetch('http://192.168.1.115:8000/analyze', {
                  method: 'POST',
                  headers: {
                      'Content-Type': 'application/json',
                  },
                  body: JSON.stringify({ imageb64: base64Data }),
              });
              
              if (!response.ok) {
                  Alert.alert('Error', `La respuesta del servidor no es correcta (Status: ${response.status})`);
                  return;
              }

              const result = await response.json();
               console.log('Respuesta del servidor recibida');

               // Guardamos los datos y mostramos el modal
               setResultData(result);
               setModalVisible(true);

            } catch (error) {
              console.error(error);
              Alert.alert('Error de conexión', 'No se pudo conectar con el servidor.');
            }
    }

    return (
        <View style={styles.container}>
            <CameraView style={styles.camera} active={recording} ref={cameraRef} />
            <View style={styles.recordingText}>
                <View style={{ backgroundColor: recording ? 'red' : 'gray', borderRadius: 4, flexDirection: 'row', paddingHorizontal: 4 }}>
                    <Fontisto name="record" size={10} color='white' style={{marginRight: 6, marginTop: 5}} />
                    <Text style={{color: 'white'}}>{recording ? 'Camera...' : 'Not Camera'}</Text>
                </View>
            </View>
            <View style={styles.buttonContainer}>
                <TouchableOpacity style={recording ? styles.button_recording : styles.button_not_recording}
                    onPress={toggleCameraRecording}>

                    <Fontisto name="record" size={24} color={recording ? 'white' : 'black'} />

                </TouchableOpacity>

                <TouchableOpacity style={recording ? styles.button_recording : styles.button_not_recording}
                    onPress={takePicture}>

                    <Fontisto name="camera" size={24} color={recording ? 'white' : 'black'} />

                </TouchableOpacity>

                <Modal visible={modalVisible} transparent={true} animationType="slide">
                    <View style={styles.modalOverlay}>
                        <View style={styles.modalContent}>
                            <Text style={styles.modalTitle}>Análisis Completado</Text>

                             {resultData && (
                                 <>
                                     <Image
                                         source={{ uri: `data:image/jpeg;base64,${resultData.imageb64}` }} // NO NOS HACE FALTA DESCODIFICAR, REACT NATIVE LO HACE AUTOMATICAMENTE

                                         style={styles.resultImage}
                                         resizeMode="contain"
                                     />
                                     <Text style={styles.modalText}>Hojas detectadas: {resultData.number}</Text>
                                 </>
                             )}

                             <TouchableOpacity
                                 onPress={() => setModalVisible(false)}
                                 style={styles.closeButton}>
                                 <Text style={styles.closeButtonText}>Cerrar</Text>
                             </TouchableOpacity>
                        </View>
                    </View>
                </Modal>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
    },
    message: {
        textAlign: 'center',
        paddingBottom: 10,
    },
    camera: {
        flex: 1,
    },
    buttonContainer: {
        position: 'absolute',
        bottom: 64,
        flexDirection: 'row',
        backgroundColor: 'transparent',
        width: '100%',
        paddingHorizontal: 64,
        justifyContent: 'center',
        alignItems: 'center',
        gap: 20
    },
    text: {
        fontSize: 24,
        fontWeight: 'bold',
        color: 'white',
    },
    button_recording: {
        backgroundColor: 'red',
        borderRadius: 50,
        padding: 30,
        alignItems:'center',
        alignSelf: 'center'
    },
    button_not_recording: {
        backgroundColor: 'white',
        borderRadius: 50,
        padding: 30,
        alignItems:'center',
        alignSelf: 'center'
    },
    recordingText: {
        position: 'absolute',
        top: Constants.statusBarHeight + 5,
        right: 0,
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 8,
        zIndex: 10,
        flexDirection: 'row',
    },
    modalOverlay: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(0,0,0,0.7)'
    },modalContent: {
        backgroundColor: 'white',
        padding: 20,
        borderRadius: 20,
        alignItems: 'center',
        width: '85%',
        maxHeight: '80%'
    },
    modalTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 15
    },
    resultImage: {
        width: '100%',
        height: 300,
        borderRadius: 10,
        marginBottom: 15,
    },
    modalText: {
        fontSize: 18,
        marginBottom: 20,
    },
    closeButton: {
        backgroundColor: '#2196F3',
        paddingHorizontal: 30,
        paddingVertical: 10,
        borderRadius: 10
    },
    closeButtonText: {
        color: 'white',
        fontWeight: 'bold'
    }
});
