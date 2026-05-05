import React, { useState } from 'react';
import {
    View, Text, TouchableOpacity, Alert, StyleSheet,
    ActivityIndicator, Image, Modal, ImageBackground,
    FlatList, ScrollView
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { IP_ADDRESS } from "@env";

const UPLOAD_URL = `http://${IP_ADDRESS}:8000/analyze`;

const RECENT_PHOTOS_MOCK = [
    {id: 1, path: 'https://imgs.search.brave.com/lqKlU24lyWlGN_rPg8nNbo5OEPP109Pw0x34pJQZ7Hk/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93d3cu/YnJpbGxhbnRlLmVz/L3dwLWNvbnRlbnQv/dXBsb2Fkcy8yMDI0/LzEyL1BpbWllbnRv/cy1sb2Rvc2EtMS5q/cGc'},
    {id: 2, path: 'https://imgs.search.brave.com/miErJhVhF_RvZdbbKDA3mhv90m7Quvs7Npyxuvu_gAQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93d3cu/aHVlcnRhYmFyYmVy/ZXRhLmNvbS8xNTMt/bGFyZ2VfZGVmYXVs/dC8yNS1wbGFudGFz/LWRlLXBpbWllbnRv/LWl0YWxpYW5vLmpw/Zw'},
    {id: 3, path: 'https://imgs.search.brave.com/jDCmLyl1-gOqhfLf11zm-LdcK1XhQId6PlIes1x_eQc/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZXJ0/aXNiZWxjaGltLmVz/L3dwLWNvbnRlbnQv/dXBsb2Fkcy8yMDIz/LzAxL1NpbnRvbWFz/X2RlX29pZGlvX2Vu/X3BsYW50YV9kZV9w/aW1pZW50by5qcGc'}
];

interface Photo {
    id: number;
    path: string;
}

interface DetectionResult {
    leaf_count: number;
    image_base64: string;
}

const UploadingScreen = ({ navigation }: any) => {
    const [loading, setLoading] = useState(false);
    const [modalVisible, setModalVisible] = useState(false);
    const [resultData, setResultData] = useState<DetectionResult | null>(null);
    const [recentPhotos, setRecentPhotos] = useState(RECENT_PHOTOS_MOCK);

    function addToRecentPhotos(newPhotoUri: Photo) {
        setRecentPhotos(prev => [newPhotoUri, ...prev.slice(0, 4)]); // Mantiene solo los 5 más recientes
    }

    const uploadFile = async (base64Data: string) => {
        try {
            setLoading(true);
            await fetchPicture(base64Data);
            console.log('Archivo enviado y procesado');
        } catch (error) {
            Alert.alert('Error', 'No se pudo subir el archivo.');
        } finally {
            setLoading(false);
        }
    }

    const uploadFileFromUri = async (uri: string) => {
        try {
            setLoading(true);
            const response = await fetch(uri);
            const blob = await response.blob();
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64data = reader.result as string;
                const base64String = base64data.split(',')[1]; // eliminamos el prefijo data:image/jpeg;base64,
                uploadFile(base64String);
            };
            reader.readAsDataURL(blob);
        }catch (error) {
            Alert.alert('Error', 'No se pudo subir el archivo desde URI.');
        }
    }

    const pickAndUploadFile = async () => {
        const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (permissionResult.granted === false) {
            Alert.alert('Permiso Requerido', 'Necesitas dar permiso para acceder a la galería.');
            return;
        }

        const pickedResult = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: false,
            quality: 1,
            base64: true,
        });

        if (pickedResult.canceled) return;

        const { base64, uri } = pickedResult.assets[0];

        if (base64) {
            addToRecentPhotos({ id: Date.now(), path: uri });
            await uploadFile(base64);
        }
    };

    async function fetchPicture(base64Data: string) {
        try {
            const response = await fetch(UPLOAD_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ imageb64: base64Data }),
            });

            if (!response.ok) throw new Error(`Status: ${response.status}`);

            const jsonRecived = await response.json();
            if (jsonRecived && typeof jsonRecived.leaf_count === 'number' && typeof jsonRecived.image_base64 === 'string') {
                const result: DetectionResult = {
                    leaf_count: jsonRecived.leaf_count,
                    image_base64: jsonRecived.image_base64,
                };
                setResultData(result);
                setModalVisible(true);
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
        <ImageBackground
            source={require('../../assets/home_background.jpg')}
            style={styles.backgroundImage}
            blurRadius={4}
        >
            <View style={styles.overlay}>
                <ScrollView contentContainerStyle={styles.content}>
                    <Text style={styles.mainTitle}>Subir Multimedia</Text>

                    <TouchableOpacity style={styles.dashedContainer} onPress={pickAndUploadFile}>
                        <View style={styles.plusCircle}>
                            <Ionicons name="add" size={50} color="white" />
                        </View>
                        <Text style={styles.dashedText}>Pulse para seleccionar la foto que se quiere analizar</Text>
                    </TouchableOpacity>

                    <View style={styles.recentSection}>
                        <Text style={styles.recentTitle}>Fotos Recientes</Text>
                        <FlatList
                            data={recentPhotos}
                            horizontal
                            showsHorizontalScrollIndicator={false}
                            keyExtractor={(item) => item.id.toString()}
                            renderItem={({ item }) => (
                                <TouchableOpacity onPress={() => uploadFileFromUri(item.path)}>
                                    <Image source={{ uri: item.path }} style={styles.recentImage} />
                                </TouchableOpacity>
                            )}
                        />
                    </View>
                </ScrollView>
            </View>
            <Modal visible={modalVisible} transparent={true} animationType="slide">
                <View style={styles.modalOverlay}>
                    <View style={styles.modalContent}>
                        <Text style={styles.modalTitle}>Análisis Completado</Text>
                        {loading ? <ActivityIndicator size="large" color="#00875A" /> : (
                            resultData && (
                                <>
                                    <Image
                                        source={{ uri: `data:image/jpeg;base64,${resultData.image_base64}` }}
                                        style={styles.resultImage}
                                        resizeMode="contain"
                                    />
                                    <Text style={styles.modalText}>Hojas detectadas: {resultData.leaf_count}</Text>
                                </>
                            )
                        )}
                        <TouchableOpacity
                            onPress={() => setModalVisible(false)}
                            style={styles.closeButton}>
                            <Text style={styles.closeButtonText}>Cerrar</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </Modal>
        </ImageBackground>
    );
};

const styles = StyleSheet.create({
    backgroundImage: { flex: 1 },
    overlay: { flex: 1, backgroundColor: 'rgba(255,255,255,0.7)' },
    headerContainer: {
        backgroundColor: '#00875A',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingVertical: 10,
    },
    backButton: { flexDirection: 'row', alignItems: 'center', paddingLeft: 10 },
    backText: { color: 'white', fontSize: 16, marginLeft: 5 },
    headerTitle: { color: 'white', fontSize: 18, fontWeight: 'bold' },
    content: { padding: 25, alignItems: 'center' },
    mainTitle: {
        fontSize: 34,
        fontWeight: 'bold',
        color: '#004D40',
        marginTop: 20,
        marginBottom: 30,
    },
    dashedContainer: {
        width: '100%',
        height: 180,
        borderWidth: 2,
        borderColor: '#004D40',
        borderStyle: 'dashed',
        borderRadius: 20,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(0, 77, 64, 0.05)',
    },
    plusCircle: {
        width: 70,
        height: 70,
        borderRadius: 35,
        backgroundColor: '#00875A',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 10,
    },
    dashedText: { color: '#004D40', fontSize: 16, fontWeight: '500' },
    recentSection: { width: '100%', marginTop: 30 },
    recentTitle: { fontSize: 20, fontWeight: 'bold', color: '#004D40', marginBottom: 15 },
    recentImage: { width: 100, height: 100, borderRadius: 15, marginRight: 15 },
    footer: { padding: 20, paddingBottom: 40 },
    uploadActionButton: {
        flexDirection: 'row',
        backgroundColor: 'white',
        borderWidth: 2,
        borderColor: '#00875A',
        paddingVertical: 15,
        borderRadius: 30,
        justifyContent: 'center',
        alignItems: 'center',
    },
    uploadActionText: { color: '#00875A', fontSize: 18, fontWeight: 'bold', marginLeft: 10 },
    modalOverlay: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.8)' },
    modalContent: { backgroundColor: 'white', padding: 20, borderRadius: 25, alignItems: 'center', width: '90%' },
    modalTitle: { fontSize: 22, fontWeight: 'bold', marginBottom: 15 },
    resultImage: { width: '100%', height: 350, borderRadius: 15, marginBottom: 15 },
    modalText: { fontSize: 18, marginBottom: 20 },
    closeButton: { backgroundColor: '#00875A', paddingHorizontal: 40, paddingVertical: 12, borderRadius: 15 },
    closeButtonText: { color: 'white', fontWeight: 'bold', fontSize: 16 },
});

export default UploadingScreen;