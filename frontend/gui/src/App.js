import React, { Component } from 'react';
import classname from 'classnames'
import './scss/index.scss';
import 'semantic-ui-css/semantic.min.css'
import AppMenu from './containers/Layout';
import FunctionsList from './containers/FunctionsList';
import {Container, Grid, Input, Image} from "semantic-ui-react";

class App extends Component {
    state = {
        image: null,
        imageCounter: 1,
        inputCounter: 1,
        path: {
            adjacency: {},
            nodes: {},
            operations: {},
            inputs: {}
        },
        disabled: true
    };


    handleFile = (e) => {
        const content = e.target.result;
        let number = this.state.imageCounter;
        let inputCount = this.state.inputCounter;
        this.setState({imageCounter: number + 1});
        this.setState({inputCounter: inputCount + 1});
        this.setState({image: content});
        this.setState({disabled: false});
        this.setState(prevState => ({
            ...prevState,
            path: {
                ...prevState.path,
                nodes: {
                    ...prevState.path.nodes,
                    ["img_" + number] : null
                },
                inputs: {
                    ...prevState.path.inputs,
                    ["input_" + inputCount]: {
                        source: content.split(',')[1],
                        to: "img_" + number
                    }
                }
            }
        }));
        console.log(content);
        console.log(this.state.path);
    };


    encodeImage = (file) => {
        let fileData = new FileReader();
        fileData.onloadend = this.handleFile;
        fileData.readAsDataURL(file)
    };

    render() {
        const style = classname('app-container');
        const gridClass = classname('app-grid');

        return (
        <div className={style}>
            <Container fluid={true}>
                <AppMenu/>
            </Container>
            <Grid
                padded
                className={gridClass}>
                <Grid.Row
                    style={ {padding: 0, 'maxHeight': '94vh'}}>
                    <Grid.Column
                        width={2}
                        style={ {'paddingLeft': 0}}>
                        <FunctionsList disabled={this.state.disabled}/>
                    </Grid.Column>
                    <Grid.Column
                        width={14}>
                        <Input type='file' accept='image/*' onChange={e => this.encodeImage(e.target.files[0])}/>
                        <Image name='test' src={this.state.image}/>
                    </Grid.Column>
                </Grid.Row>
            </Grid>
        </div>
        );
    }
}

export default App;
