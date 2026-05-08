import React, { useEffect, useState, useRef } from 'react';
import {
    View, Text, TouchableOpacity, Alert, StyleSheet,
    ActivityIndicator, Image, Modal,
    FlatList, ScrollView, Dimensions
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { IP_ADDRESS } from "@env";
import { useSession } from '../SessionContext';
import AppBackground from '../component/AppBackground';

const UPLOAD_URL = `http://${IP_ADDRESS}:8000/analyze`;


interface DetectionResult {
    leaf_count: number;
    image_base64: string;
}

const UploadingScreen = ({ navigation }: any) => {
    const { recents, setRecents, user } = useSession();
    const [loading, setLoading] = useState(false);
    const [modalVisible, setModalVisible] = useState(false);
    const [resultData, setResultData] = useState<DetectionResult | null>(null);
    const [recentPhotos, setRecentPhotos] = useState<Base64URLString[]>([]);
    const [galleryModalVisible, setGalleryModalVisible] = useState(false);
    const [selectedPhotoIndex, setSelectedPhotoIndex] = useState(0);
    const flatListRef = useRef<FlatList<any> | null>(null);
    const { width: windowWidth, height: windowHeight } = Dimensions.get('window');

    useEffect(() => {
        setRecentPhotos(recents);
    }, [recents]);

    function addPhotoToRecents(newPhotoBase64: Base64URLString) {
        const newPhoto: Base64URLString = newPhotoBase64;
        setRecents((prev: [Base64URLString] | any[]) => [newPhoto, ...prev.slice(0, 4)]); // Mantiene solo los 5 más recientes
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
            await uploadFile(base64);
        }
    };

    async function fetchPicture(base64Data: string) {
        try {
            const response = await fetch(UPLOAD_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ imageb64: base64Data, user_id: user?.id }),
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
        <AppBackground>
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
                            keyExtractor={(item, index) => String(index)}
                            renderItem={({ item, index }) => (
                                <TouchableOpacity onPress={() => {
                                    setSelectedPhotoIndex(index);
                                    setGalleryModalVisible(true);
                                }}>
                                    <Image source={{ uri: `data:image/jpeg;base64,${item}` }} style={styles.recentImage} />
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
                        {loading ? (
                            <View style={styles.loadingWrapper}>
                                <View style={styles.innerCircle}>
                                    <ActivityIndicator size="large" color="#00E676" />
                                </View>
                                <Text style={styles.loadingTitle}>Analizando imagen...</Text>
                                <Text style={styles.loadingSubtext}>Esto puede tardar unos segundos. Gracias por tu paciencia.</Text>
                            </View>
                        ) : (
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
            <Modal visible={galleryModalVisible} transparent={true} animationType="fade">
                <View style={styles.galleryOverlay}>
                    <FlatList
                        ref={(ref) => { flatListRef.current = ref as any; }}
                        data={recentPhotos}
                        horizontal
                        pagingEnabled
                        initialScrollIndex={selectedPhotoIndex}
                        getItemLayout={(_, index) => ({ length: windowWidth, offset: windowWidth * index, index })}
                        keyExtractor={(item, index) => String(index)}
                        renderItem={({ item, index }) => (
                            <View style={[styles.galleryItem, { width: windowWidth, height: windowHeight }]}> 
                                <Image
                                    source={{ uri: `data:image/jpeg;base64,${item}` }}
                                    style={[styles.fullImage, { width: windowWidth, height: windowHeight * 0.85 }]}
                                    resizeMode="contain"
                                />
                            </View>
                        )}
                    />

                    <TouchableOpacity style={styles.galleryClose} onPress={() => setGalleryModalVisible(false)}>
                        <Ionicons name="close" size={34} color="#fff" />
                    </TouchableOpacity>

                    <View style={styles.galleryNavRow} pointerEvents="box-none">
                        <TouchableOpacity style={styles.galleryNavButton} onPress={() => {
                            const prev = Math.max(0, selectedPhotoIndex - 1);
                            setSelectedPhotoIndex(prev);
                            flatListRef.current?.scrollToIndex({ index: prev, animated: true });
                        }}>
                            <Ionicons name="chevron-back" size={36} color="#fff" />
                        </TouchableOpacity>

                        <TouchableOpacity style={styles.galleryNavButton} onPress={() => {
                            const next = Math.min(recentPhotos.length - 1, selectedPhotoIndex + 1);
                            setSelectedPhotoIndex(next);
                            flatListRef.current?.scrollToIndex({ index: next, animated: true });
                        }}>
                            <Ionicons name="chevron-forward" size={36} color="#fff" />
                        </TouchableOpacity>
                    </View>
                </View>
            </Modal>
            <Modal visible={loading} transparent={true} animationType="fade">
                <View style={styles.modalOverlay}>
                    <View style={styles.modalContent}>
                        <View style={styles.innerCircle}>
                            <ActivityIndicator size="large" color="#00E676" />
                        </View>
                        <Text style={styles.loadingTitle}>Analizando imagen...</Text>
                        <Text style={styles.loadingSubtext}>Esto puede tardar unos segundos. Gracias por tu paciencia.</Text>
                    </View>
                </View>
            </Modal>
        </AppBackground>
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
    dashedText: { color: '#004D40', fontSize: 16, fontWeight: '500', textAlign: 'center' },
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
    innerCircle: {
        width: 88,
        height: 88,
        borderRadius: 44,
        backgroundColor: 'white',
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: '#00E676',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.12,
        shadowRadius: 12,
        elevation: 6,
        marginBottom: 12,
    },
    loadingTitle: {
        fontSize: 18,
        fontWeight: '700',
        marginTop: 6,
        color: '#222',
    },
    loadingSubtext: {
        fontSize: 13,
        color: '#666',
        marginTop: 6,
        textAlign: 'center',
    },
    loadingWrapper: { alignItems: 'center' },
    galleryOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.95)', justifyContent: 'center', alignItems: 'center' },
    galleryItem: { justifyContent: 'center', alignItems: 'center' },
    fullImage: { borderRadius: 12 },
    galleryClose: { position: 'absolute', top: 48, right: 20, zIndex: 20 },
    galleryNavRow: { position: 'absolute', bottom: 40, left: 0, right: 0, flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 30, alignItems: 'center' },
    galleryNavButton: { backgroundColor: 'rgba(0,0,0,0.35)', padding: 10, borderRadius: 30 },
});

export default UploadingScreen;