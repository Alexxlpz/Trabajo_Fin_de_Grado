import React from 'react';
import Constants from 'expo-constants';
import {Text, View} from "react-native";

const Main = () => {
    return (
        <View style={{ marginTop: Constants.statusBarHeight, flexGrow: 1 }}>
            <Text>Main Comonrnt</Text>
        </View>
    );
}

export default Main;