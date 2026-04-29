import React, { useState } from 'react';
import {View, Text, TouchableOpacity, Alert, StyleSheet, ActivityIndicator, Image, Modal} from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const UPLOAD_URL = 'http://192.168.1.116:8000/analyze';

const UploadComponent = () => {
    const [loading, setLoading] = useState(false);

    const [modalVisible, setModalVisible] = useState(false);
    const [resultData, setResultData] = useState<{number: number, imageb64: string} | null>(null);

    const pickAndUploadFile = async () => {
        // pedimos permisos al usuario
        const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (permissionResult.granted === false) {
            Alert.alert('Permiso Requerido', 'Necesitas dar permiso para acceder a la galería de fotos.');
            return;
        }

        // escogemos el archivo
        const pickedResult = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.All,
            allowsEditing: false,
            quality: 1,
            base64: true,
        });

        if (pickedResult.canceled) {
            return;
        }

        const { base64 } = pickedResult.assets[0];

        // probamos a subir el archivo
        try {
            setLoading(true);

            if (base64) {
                await fetchPicture(base64);
                console.log('Imagen enviada al servidor');
            } else {
                Alert.alert('Error', 'No se pudo procesar la imagen seleccionada.');
            }

        } catch (error) {
            console.error("Error al subir el archivo:", error);
            Alert.alert('Error', 'No se pudo subir el archivo.');
        } finally {
            setLoading(false);
        }
    };

    async function fetchPicture(base64Data: string){
            try {
                // lo enviamos en una peticion post ya que es exageradamente larga la cadena de b64
              const response = await fetch(UPLOAD_URL, {
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
        <Text style={styles.header}>Subir Multimedia</Text>
        <TouchableOpacity
            style={styles.uploadButton}
            onPress={pickAndUploadFile}
            disabled={loading}
        >
            {loading ? (
                    <ActivityIndicator size="large" color="#FFF" />
                    ) : (
                    <Text style={styles.buttonText}>+</Text>
                    )}
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
        );
    };

    const styles = StyleSheet.create({
        container: {
            flex: 1,
            justifyContent: 'center',
            alignItems: 'center',
            padding: 20,
            backgroundColor: '#f5f5f5',
        },
        header: {
            fontSize: 24,
            fontWeight: 'bold',
            marginBottom: 20,
        },
        uploadButton: {
            backgroundColor: '#007AFF', // Azul para el botón de acción
            width: 80,
            height: 80,
            borderRadius: 40,
            justifyContent: 'center',
            alignItems: 'center',
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 4 },
            shadowOpacity: 0.3,
            shadowRadius: 5,
            elevation: 8,
        },
        buttonText: {
            color: '#FFF',
            fontSize: 40,
            lineHeight: 40,
        },
        fileType: {
            marginTop: 20,
            fontSize: 14,
            color: '#666',
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

    export default UploadComponent;