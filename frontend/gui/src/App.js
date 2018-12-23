import React, { Component } from 'react';
import classname from 'classnames'
import './scss/index.scss';
import 'semantic-ui-css/semantic.min.css'
import AppMenu from './components/Layout';
import FunctionsList from './containers/FunctionsList';
import {Button, Container, Dimmer, Grid, Icon, Image, Loader, Modal} from "semantic-ui-react";
import axios from "axios";

class App extends Component {

    constructor(props) {
        super(props);
        this.svg_ref = React.createRef();
    }

    state = {
        image: null,
        activeItem: null,
        activeDimmer: false,
        imageCounter: 1,
        inputCounter: 1,
        operationCounter: 1,
        path: {
            adjacency: {},
            nodes: {},
            operations: {},
            inputs: {}
        },
        disabled: true,
        children: [],
        imgs: [],
        style: {
            height: "1000%"
        },
        modalOpen: false
    };


    handleFile = (e) => {
        const content = e.target.result;
        let number = this.state.imageCounter - 1;
        let inputCount = this.state.inputCounter - 1;
        this.setState(prevState => ({
            ...prevState,
            path: {
                ...prevState.path,
                nodes: {
                    ...prevState.path.nodes,
                    ["img_" + number]  : null
                },
                inputs: {
                    ...prevState.path.inputs,
                    ["input_" + inputCount]: {
                        source: content.split(',')[1],
                        to: "img_" + number
                    }
                }
            },
            disabled: false
        }));
    };


    encodeImage = (e) => {
        let file = e.target.files[0];
        let name = file.name;
        let fileData = new FileReader();
        let number = this.state.imageCounter;
        let inputCount = this.state.inputCounter;
        fileData.onloadend = this.handleFile;
        fileData.readAsDataURL(file);
        this.setState(prevState => ({
            ...prevState,
                imageCounter: number + 1,
                inputCounter: inputCount + 1,
                imgs: [...prevState.imgs,
                    <div className={"input_image_div"} key={["img_" + number]} style={{top:[(inputCount-1)*150 + 25 +"px"]}}>{name}</div>],
        }));
    };

     test = (e) => {
         if (this.state.activeItem != null) {
             let operationName = this.state.activeItem;
             let number = this.state.imageCounter;
             let operationCount = this.state.operationCounter;
             let lastimg = ("img_"+ number -1);
             let state =  this.state.path.adjacency[lastimg];
             const xPosition = e.pageX - e.target.offsetLeft + e.target.scrollLeft;
             const yPosition = e.pageY - e.target.offsetTop + e.target.scrollTop;
             console.log(xPosition , yPosition);
             this.setState(prevState => ({
                 ...prevState,
                 activeItem: null,
                 imageCounter: number + 1,
                 operationCounter: operationCount + 1,
                 path: {
                     ...prevState.path,
                     adjacency: {
                         ...prevState.path.adjacency,
                         ["img_" + (number - 1)]: ["img_" + number]
                     },
                     nodes: {
                         ...prevState.path.nodes,
                         ["img_" + number]: null
                     },
                     operations: {
                         ...prevState.path.operations,
                         ["operation_" + operationCount]: {
                             operation_name: operationName,
                             from: ["img_" + (number - 1)],
                             to: "img_" + number,
                             "params": []
                         }
                     }
                 },
                 disabled: false,
                 imgs: [...prevState.imgs,
                    <div className={this.state.activeItem + "_image_div"} key={["img_" + number]}
                         style={{
                             display: 'flex',
                             top:[yPosition  - 25 +"px"],
                             position: 'absolute',
                             left:[xPosition - 50 +"px"],
                             width: "100px",
                             height: '50px',
                             border: "1px solid black",
                             justifyContent: 'center',
                             alignItems: 'center'
                         }}>{this.state.activeItem}</div>],
             }));
         }
         // dużo if w zależności od rodzaju obrazka xD + on click handler dla każdego( ta sama operacja prawie( jakieś menu czy coś)
         // this.setState(prevState => ({
         //     imgs: [...prevState.imgs,
         //            <div className={"input_image_div"} key={["img_" + number]} style={{top:[(number-1)*150 + 25 +"px"]}}>{name}</div>],
         // }));
    };

     handleScroll = (e) => {
         this.svg_ref.current.scrollTop = e.target.scrollTop;
     };

     handleItemClick = (e, { name }) => {
        this.setState({ activeItem: name });
        console.log(name)
    };

     sendPath = (e) => {
         this.setState({activeDimmer: true});
         axios.post('http://127.0.0.1:8000/process/',{
                path: this.state.path
         })
            .then( res =>{
                this.setState({activeDimmer: false, modalOpen: true, image: "data:image/jpeg;base64," + res.data.ret });
            })
    };
     handleClose = () => {
         this.setState({ modalOpen: false })
     };

    render() {
        const style = classname('app-container');
        const gridClass = classname('app-grid');

        return (
        <div className={style}>
            <Container fluid={true}>
                <AppMenu
                    imagesHandler={this.encodeImage}
                    handleSend={this.sendPath}
                />
            </Container>
            <Container fluid={true}>
            <Grid
                padded
                className={gridClass}
                >
                    <Grid.Column
                        width={2}
                        style={ {padding: 0}}
                    >
                        <FunctionsList
                            disabled={this.state.disabled}
                            active={this.state.activeItem}
                            handleItemClick={this.handleItemClick}/>
                    </Grid.Column>
                    <Grid.Column
                        width={14}
                        style={ {padding: 0}}
                    >
                        <div ref={this.svg_ref} style={{position: 'fixed', zIndex:1, width:"100%", height:"94vh", overflowY:'scroll'}}>
                            <svg id={"test"} style={this.state.style}>
                                <defs>
                                    <marker id='head' orient="auto"
                                            markerWidth='4' markerHeight='8'
                                            refX='0.1' refY='4'>
                                        <path d='M0,0 V8 L4,4 Z' fill="red"/>
                                    </marker>
                                </defs>
                                {this.state.children}
                            </svg>

                        </div>
                        <div onScroll={this.handleScroll} onClick={this.test} style={{position: 'fixed', zIndex:999, width:"100%", height:"94vh",overflowY:'scroll'}}>
                            {this.state.imgs}
                        </div>
                    </Grid.Column>
            </Grid>
            </Container>
            <Dimmer active={this.state.activeDimmer}>
                <Loader indeterminate>Trwa przetwarzanie obrazu</Loader>
            </Dimmer>
            <Modal
                open={this.state.modalOpen}
                onClose={this.handleClose}
            >
                <Modal.Content>
                   <Image name='test' src={this.state.image}/>
                </Modal.Content>
                <Modal.Actions>
                    <Button color='green' onClick={this.handleClose} inverted>
                        <Icon name='checkmark' /> Got it
                    </Button>
                </Modal.Actions>
            </Modal>
        </div>
        );
    }
}

/*children: [...prevState.children,
<line y1={(number-1)*150 + 25} x1={25} y2={(number-1)*150 + 125} x2={125} markerEnd={ 'url(#head)' }  style={{ stroke: 'black', strokeWidth: 2 }} />]*/

export default App;
