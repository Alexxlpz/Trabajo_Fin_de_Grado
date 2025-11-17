import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Alert, StyleSheet, ActivityIndicator } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

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

        // funcion para escoger el archivo
        const pickerResult = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.All, // Permite fotos y videos
            allowsEditing: true,
            aspect: [4, 3],
            quality: 1,
        });

        if (pickerResult.canceled) {
            setStatus('Selección cancelada.');
            return;
        }

        const { uri, type } = pickerResult.assets[0];
        setFileType(type);

        // probamos a subir el archivo
        try {
            setLoading(true);
            setStatus(`Subiendo ${type}...`);

            //const downloadURL = await uploadFile(uri, type);

            //setStatus(`¡Subida exitosa! URL: ${downloadURL.substring(0, 40)}...`);
            Alert.alert('Éxito', 'El archivo ha sido subido a Firebase Storage.');

        } catch (error) {
            console.error("Error al subir el archivo:", error);
            setStatus('Error al subir. Revisa la consola.');
            Alert.alert('Error', 'No se pudo subir el archivo. Consulta los detalles en la consola.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <View style={styles.container}>
        <Text style={styles.header}>Subir Multimedia a Firebase</Text>
    <Text style={styles.status}>{status}</Text>

    {/* Botón "+" para iniciar la subida */}
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

    {/* Indicador de qué tipo de archivo se seleccionó */}
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