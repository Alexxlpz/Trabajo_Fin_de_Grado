import React from 'react';
import { Text, View, StyleSheet, Button } from "react-native";

const HomeScreen = ({ navigation }) => {
    return (
        <View style={styles.container}>
            <Text style={styles.title}> titulo</Text>
            <Button
                title="Grabar Video"
                onPress={() => navigation.navigate('Recording')}
            />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        gap: 10
    },
    title: {
        fontSize: 20
    }
});

export default HomeScreen;
