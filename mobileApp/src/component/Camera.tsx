import { CameraView, useCameraPermissions } from 'expo-camera';
import {useState, useRef, useEffect} from 'react';
import {
    Modal,
    Button,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
    Alert,
    Image,
    Platform,
    StatusBar
} from 'react-native';
import { Fontisto } from '@expo/vector-icons';
import Constants from "expo-constants";
import { IP_ADDRESS } from "@env";
import { useSession } from '../SessionContext';

interface DetectionResult {
    leaf_count: number;
    image_base64: string;
}

export default function Camera() {
    const { recents, setRecents } = useSession();
    const [recording, setRecording] = useState<boolean>(false);
    const [modalVisible, setModalVisible] = useState(false);
    const [resultData, setResultData] = useState<DetectionResult | null>(null); // guardamos el return que nos devuelve el backend, que contiene el numero de objetos detectados y la imagen en b64
    const [cameraMode, setCameraMode] = useState<'FOTO' | 'VIDEO'>('FOTO');
    const [isLoading, setLoading] = useState(false);

    const cameraRef = useRef<CameraView>(null);
    const recordingRef = useRef<boolean>(false);
    const isLoadingRef = useRef<boolean>(false);
    const modalVisibleRef = useRef<boolean>(false);
    const [permission, requestPermission] = useCameraPermissions();


    useEffect(() => { isLoadingRef.current = isLoading; }, [isLoading]);
    useEffect(() => { modalVisibleRef.current = modalVisible; }, [modalVisible]);

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

    function addPhotoToRecents(newPhotoBase64: Base64URLString) {
        const newPhoto: Base64URLString = newPhotoBase64;
        setRecents((prev: [Base64URLString] | any[]) => [newPhoto, ...prev.slice(0, 4)]); // Mantiene solo los 5 más recientes
    }

    function toggleCameraRecording(){
        setRecording(current => {
            const next = !current;
            recordingRef.current = next;
            if (next) {
                performVideo();
            }
            return next;
        });
    }

    async function performVideo(){
        const sleep = (ms: number) => new Promise(res => setTimeout(res, ms));

        while (recordingRef.current) {
            if (!isLoadingRef.current && !modalVisibleRef.current) {
                await takePicture();

                if (!recordingRef.current) { //para que no me de la respuesta si lo paro antes de que me llegue
                    setResultData(null);
                    setModalVisible(false);
                }

                await sleep(100);
            } else {
                await sleep(100);
            }
        }
    }

    async function takePicture(){
        setLoading(true);
         if (cameraRef.current) {
             try {
                 const options = { quality: 1, base64: true };
                 const data = await cameraRef.current.takePictureAsync(options);
                 console.log('Imagen capturada:');

                 // si el dato no es null se lo pasamos a la función para que lo mande al backend y lo analice
                 //console.log(data);
                 if (data !== null) {
                     //console.log('antes de enviar la imagen');
                     await fetchPicture(data.base64!);
                     setLoading(false);
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
              const response = await fetch(`http://${IP_ADDRESS}:8000/analyze`, {
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

              const jsonRecived = await response.json();
               console.log('Respuesta del servidor recibida');

               // Guardamos los datos y mostramos el modal
                if (jsonRecived && typeof jsonRecived.leaf_count === 'number' && typeof jsonRecived.image_base64 === 'string') {
                    const result: DetectionResult = {
                        leaf_count: jsonRecived.leaf_count,
                        image_base64: jsonRecived.image_base64,
                    };
                    setResultData(result);
                    setModalVisible(true);
                    addPhotoToRecents(result.image_base64);
                } else {
                    Alert.alert('Error', 'Respuesta inesperada del servidor');
                    console.error('Respuesta inválida del servidor:', jsonRecived);
                }
            } catch (error) {
              console.error(error);
              Alert.alert('Error de conexión', 'No se pudo conectar con el servidor.');
            }
    }

    return (
        <View style={styles.container}>
            {Platform.OS === 'ios' && <StatusBar barStyle="light-content" />}

            <CameraView
                style={styles.camera}
                ref={cameraRef}
                mode={cameraMode === 'FOTO' ? 'picture' : 'video'}
            />
            <View style={styles.recordingText}>
                {cameraMode === 'VIDEO' && (
                    <View style={{ backgroundColor: recording ? 'red' : 'gray', borderRadius: 4, flexDirection: 'row', paddingHorizontal: 4 }}>
                        <Fontisto name="record" size={10} color='white' style={{marginRight: 6, marginTop: 5}} />
                        <Text style={{color: 'white'}}>{recording ? 'Recording...' : 'Pause'}</Text>
                    </View>
                )}

            </View>
            <View style={styles.footer}>
                <View style={styles.selectorContainer}>
                        <TouchableOpacity onPress={() => setCameraMode('FOTO')}>
                            <Text style={[styles.modeText, cameraMode === 'FOTO' ? styles.activeMode : null]}>
                                FOTO
                            </Text>
                        </TouchableOpacity>
                        <TouchableOpacity onPress={() => {
                            setCameraMode('VIDEO');
                            setRecording(false);
                        }}>
                            <Text style={[styles.modeText, cameraMode === 'VIDEO' ? styles.activeMode : null]}>
                                VIDEO
                            </Text>
                        </TouchableOpacity>
                </View>
                {cameraMode === 'VIDEO'?  (
                    <TouchableOpacity style={recording ? styles.button_recording : styles.button_not_recording}
                                      onPress={toggleCameraRecording}>

                        <Fontisto name="record" size={24} color={recording ? 'white' : 'black'} />

                    </TouchableOpacity>
                ) : (
                    <TouchableOpacity style={styles.shutterOuter} onPress={takePicture}>
                        <View style={styles.shutterInner} />
                    </TouchableOpacity>
                )}

                <Modal visible={modalVisible} transparent={true} animationType="slide">
                    <View style={styles.modalOverlay}>
                        <View style={styles.modalContent}>
                            <Text style={styles.modalTitle}>Análisis Completado</Text>

                             {resultData && (
                                 <>
                                     <Image
                                         source={{ uri: `data:image/jpeg;base64,${resultData.image_base64}` }} // NO NOS HACE FALTA DESCODIFICAR, REACT NATIVE LO HACE AUTOMATICAMENTE

                                         style={styles.resultImage}
                                         resizeMode="contain"
                                     />
                                     <Text style={styles.modalText}>Hojas detectadas: {resultData.leaf_count}</Text>
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
    },
    footer: {
        backgroundColor: 'black',
        height: 180,
        alignItems: 'center',
        justifyContent: 'center',
    },
    selectorContainer: {
        flexDirection: 'row',
        gap: 30,
        marginBottom: 20,
    },
    modeText: {
        color: '#888',
        fontSize: 14,
        fontWeight: 'bold',
        letterSpacing: 1,
    },
    activeMode: {
        color: '#00E676', // Verde brillante para el modo activo
    },
    shutterOuter: {
        width: 80,
        height: 80,
        borderRadius: 40,
        borderWidth: 4,
        borderColor: 'white',
        justifyContent: 'center',
        alignItems: 'center',
    },
    shutterInner: {
        width: 66,
        height: 66,
        borderRadius: 33,
        backgroundColor: 'white',
    },
});
