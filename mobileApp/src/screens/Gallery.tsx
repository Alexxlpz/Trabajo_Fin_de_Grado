import React, {useState, useRef, useEffect} from 'react';
import {View, Text, StyleSheet, FlatList, Image, TouchableOpacity, Modal, Dimensions} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useSession } from '../SessionContext';

const {width} = Dimensions.get('window');
const numColumns = 3;
const padding = 8;

const Gallery: React.FC = () => {
  const { recents } = useSession();
  const [visible, setVisible] = useState(false);
  const [current, setCurrent] = useState<number>(0);
  const flatListRef = useRef<FlatList<any> | null>(null);
  const { width: windowWidth, height: windowHeight } = Dimensions.get('window');

  const open = (index: number) => {
    setCurrent(index);
    setVisible(true);
  };

  useEffect(() => {
    if (visible) {
      setTimeout(() => {
        flatListRef.current?.scrollToIndex({ index: current, animated: false });
      }, 50);
    }
  }, [visible, current]);

  const renderItem = ({item, index}: {item: string; index: number}) => {
    const uri = `data:image/jpeg;base64,${item}`;
    const size = (width - padding * (numColumns + 1)) / numColumns;

    return (
      <TouchableOpacity onPress={() => open(index)} style={[styles.thumbWrapper, {width: size, height: size}]}
                        activeOpacity={0.8}>
        <Image source={{uri}} style={styles.thumb} />
      </TouchableOpacity>
    );
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Galería</Text>
      {recents.length === 0 ? (
        <View style={styles.empty}><Text style={styles.emptyText}>No hay fotos recientes</Text></View>
      ) : (
        <FlatList
          data={recents}
          keyExtractor={(_, i) => String(i)}
          renderItem={renderItem}
          numColumns={numColumns}
          contentContainerStyle={styles.list}
        />
      )}

      <Modal visible={visible} transparent animationType="fade">
        <View style={styles.galleryOverlay}>
          <FlatList
            ref={(ref) => { flatListRef.current = ref as any; }}
            data={recents}
            horizontal
            pagingEnabled
            initialScrollIndex={current}
            getItemLayout={(_, index) => ({ length: windowWidth, offset: windowWidth * index, index })}
            keyExtractor={(item, index) => String(index)}
            renderItem={({ item}) => (
              <View style={[styles.galleryItem, { width: windowWidth, height: windowHeight }]}> 
                <Image
                  source={{ uri: `data:image/jpeg;base64,${item}` }}
                  style={[styles.fullImage, { width: windowWidth, height: windowHeight * 0.85 }]}
                  resizeMode="contain"
                />
              </View>
            )}
          />

          <TouchableOpacity style={styles.galleryClose} onPress={() => setVisible(false)}>
            <Ionicons name="close" size={34} color="#fff" />
          </TouchableOpacity>

          <View style={styles.galleryNavRow} pointerEvents="box-none">
            <TouchableOpacity style={styles.galleryNavButton} onPress={() => {
                const prev = Math.max(0, current - 1);
                setCurrent(prev);
                flatListRef.current?.scrollToIndex({ index: prev, animated: true });
            }}>
              <Ionicons name="chevron-back" size={36} color="#fff" />
            </TouchableOpacity>

            <TouchableOpacity style={styles.galleryNavButton} onPress={() => {
                const next = Math.min(recents.length - 1, current + 1);
                setCurrent(next);
                flatListRef.current?.scrollToIndex({ index: next, animated: true });
            }}>
              <Ionicons name="chevron-forward" size={36} color="#fff" />
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
    backgroundColor: '#0f1724', 
    paddingTop: 20
  },
  title: {
    color: '#fff', 
    fontSize: 22, 
    fontWeight: '600', 
    paddingHorizontal: 16, 
    marginBottom: 12
  },
  list: {
    paddingHorizontal: padding, 
    paddingBottom: 20
  },
  thumbWrapper: {
    margin: padding / 2, 
    borderRadius: 8, 
    overflow: 'hidden', 
    backgroundColor: '#0b1220'
  },
  thumb: {
    width: '100%', 
    height: '100%'
  },
  empty: {
    flex: 1, 
    justifyContent: 'center', 
    alignItems: 'center', 
    padding: 40
  },
  emptyText: {
    color: '#9aa4b2'
  },
  modalBackground: {
    flex: 1, 
    backgroundColor: 'rgba(0,0,0,0.85)', 
    justifyContent: 'center'
  },
  modalContent: {
    alignItems: 'center', 
    justifyContent: 'center'
  },
  fullImage: { 
    borderRadius: 12 
  },
  closeArea: {
    flex: 1
  },
  galleryOverlay: { 
    flex: 1, 
    backgroundColor: 'rgba(0,0,0,0.95)', 
    justifyContent: 'center', 
    alignItems: 'center' 
  },
  galleryItem: { 
    justifyContent: 'center', 
    alignItems: 'center' 
  },
  galleryClose: { 
    position: 'absolute', 
    top: 48, 
    right: 20, 
    zIndex: 20 
  },
  galleryNavRow: { 
    position: 'absolute', 
    bottom: 40, 
    left: 0, 
    right: 0, 
    flexDirection: 'row', 
    justifyContent: 'space-between', 
    paddingHorizontal: 30, 
    alignItems: 'center' 
  },
  galleryNavButton: { 
    backgroundColor: 'rgba(0,0,0,0.35)', 
    padding: 10, 
    borderRadius: 30 
  },
});

export default Gallery;
