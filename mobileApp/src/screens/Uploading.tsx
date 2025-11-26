import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Alert, StyleSheet, ActivityIndicator } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const BASE_URL = 'http://192.168.1.114:8000/v1/drive';
const UPLOAD_ENDPOINT = '/upload';

const UploadComponent = () => {
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState('Listo para subir.');
    const [fileType, setFileType] = useState('');

    const pickAndUploadFile = async () => {
        // pedimos permisos al usuario
        const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (permissionResult.granted === false) {
            Alert.alert('Permiso Requerido', 'Necesitas dar permiso para acceder a la galería de fotos.');
            return;
        }

        // escogemos el archivo
        const pickedResult = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.All, // Permite fotos y videos
            allowsEditing: true,
            quality: 1,
        });

        if (pickedResult.canceled) {
            setStatus('Selección cancelada.');
            return;
        }

        const { uri, type } = pickedResult.assets[0];
        setFileType(type);

        // probamos a subir el archivo
        try {
            setLoading(true);
            setStatus(`Subiendo ${type}...`);

            const downloadURL = await uploadFile(uri);

            setStatus(`¡Subida exitosa! URL: ${downloadURL}...`);
            Alert.alert('Éxito', 'El archivo ha sido subido correctamente.');

        } catch (error) {
            console.error("Error al subir el archivo:", error);
            setStatus('Error al subir.');
            Alert.alert('Error', 'No se pudo subir el archivo.');
        } finally {
            setLoading(false);
        }
    };
    const uploadFile = async (localUri:string) => {

        let filename = localUri.split('/').pop();
        let typeMatch = /\.(\w+)$/.exec(filename);
        let type = typeMatch ? `${fileType}/${typeMatch[1]}` : fileType;

        const formData = new FormData();
        formData.append('file', {
            uri: localUri,
            name: filename,
            type: type // (ej: 'image/jpeg' o 'video/mp4')
        } as any);

        const fullUrl = `${BASE_URL}${UPLOAD_ENDPOINT}`;

        try {
            const response = await fetch(fullUrl, {
                method: 'POST',
                body: formData,//El tipo lo detecta fastapi automáticamente
            });

            const result = await response.json();
            console.log(result);

            if (response.ok) {
                console.log("Subida exitosa:", result);
                alert("Archivo subido con éxito!");
            } else {
                console.error("Error del servidor:", result);
                alert("Error al subir el archivo.");
            }

        } catch (error) {
            console.error("Error de red/conexión:", error);
            alert("No se pudo conectar al servidor.");
        }
    };

    return (
        <View style={styles.container}>
        <Text style={styles.header}>Subir Multimedia</Text>
        <Text style={styles.status}>{status}</Text>
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

        {fileType ? <Text style={styles.fileType}>Tipo seleccionado: {fileType}</Text> : null}
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
        status: {
            marginBottom: 30,
            color: '#333',
            textAlign: 'center',
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
        }
    });

    export default UploadComponent;