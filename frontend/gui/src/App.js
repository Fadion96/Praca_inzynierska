import React, {Component} from 'react';
import classname from 'classnames'
import './scss/index.scss';
import 'semantic-ui-css/semantic.min.css'
import AppMenu from './components/Layout';
import FunctionsList from './containers/FunctionsList';
import {
    Button,
    Container,
    Dimmer, Form,
    Grid,
    Icon,
    Image,
    Loader,
    Modal,
} from "semantic-ui-react";
import axios from "axios";

class App extends Component {
    constructor(props) {
        super(props);
        this.svg_ref = React.createRef();
    }

    componentDidMount() {
        axios.get('http://127.0.0.1:8000/api/')
            .then(res => {
                this.setState({
                    functions: res.data
                });
                console.log(res.data);
            })
    }

    state = {
        image: null,
        activeItem: null,
        activeDimmer: false,
        functions: [],
        imageCounter: 1,
        inputCounter: 1,
        operationCounter: 1,
        lineCounter: 1,
        path: {
            adjacency: {},
            nodes: {},
            operations: {},
            inputs: {}
        },
        disabled: true,
        lines: [],
        imgs: [],
        centre: [],
        style: {
            height: "1000%"
        },
        modalOpen: false,
        modal2Open: false,
        modalForm: null
    };

    displayDivs = (divList) => divList.map((el) => (
        el.div
    ));

    displayLines = (linesList) => linesList.map((el) => (
        el.line
    ));

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
                    ["img_" + number]: null
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

    changeImage = (input, e) => {
        let file = e.target.files[0];
        let fileData = new FileReader();
        fileData.onloadend = (ev) => {
            const content = ev.target.result;
            input.source = content.split(',')[1];
        };
        fileData.readAsDataURL(file);
        this.handle2Close();
    };


    encodeImage = (e) => {
        let file = e.target.files[0];
        let fileData = new FileReader();
        let number = this.state.imageCounter;
        let inputCount = this.state.inputCounter;
        fileData.onloadend = this.handleFile;
        fileData.readAsDataURL(file);
        let image = {};
        image.name = "img_" + number;
        image.div =
            <div
                className={"input_image_div"}
                key={["img_" + number]}
                id={["img_" + number]}
                onClick={this.processClick}
                style={{
                    top: [(inputCount - 1) * 150 + 25 + "px"]
                }}>
                {"Obraz " + number}
            </div>;
        image.x = 75;
        image.y = (inputCount - 1) * 150 + 50;
        this.setState(prevState => ({
            ...prevState,
            imageCounter: number + 1,
            inputCounter: inputCount + 1,
            imgs: [...prevState.imgs, image],
        }));
    };

    getDivCenter = (name) => {
        return this.state.imgs.find(el => el.name === name);
    };

    getFunction = (funct) => {
        return this.state.functions.find(el => el.name === funct)
    };

    getFunctionByFunction = (funct) => {
        return this.state.functions.find(el => el.function === funct)
    };

    getOperation = (img_key) => {
        let operations = this.state.path.operations;
        for (let op in operations) {
            if (operations[op].to === img_key) {
                return operations[op]
            }
        }
    };
    getInput = (img_key) => {
        let inputs = this.state.path.inputs;
        for (let input in inputs) {
            if (inputs[input].to === img_key) {
                return inputs[input]
            }
        }
    };

    addProcess = (e) => {
        if (this.state.activeItem != null) {
            let operation = this.getFunction(this.state.activeItem);
            let number = this.state.imageCounter;
            let operationCount = this.state.operationCounter;
            const xPosition = e.pageX - e.target.offsetLeft + e.target.scrollLeft;
            const yPosition = e.pageY - e.target.offsetTop + e.target.scrollTop;

            let image = {};
            image.name = "img_" + number;
            image.div =
                <div
                    className={operation.function + "_image_div"}
                    key={["img_" + number]}
                    id={["img_" + number]}
                    onClick={this.processClick}
                    style={{
                        top: [yPosition - 25 + "px"],
                        left: [xPosition - 50 + "px"]
                    }}>
                    {operation.name}
                </div>;
            image.x = xPosition;
            image.y = yPosition;
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
                            operation_name: operation.function,
                            from: ["img_" + (number - 1)],
                            to: "img_" + number,
                            "params": JSON.parse(operation.params)
                        }
                    }
                },
                disabled: false,
                imgs: [...prevState.imgs, image],
            }), () => {
                this.fixLines();
            });
        }
    };

    handleScroll = (e) => {
        this.svg_ref.current.scrollTop = e.target.scrollTop;
    };

    handleItemClick = (e, {name}) => {
        this.setState({activeItem: name});
    };

    sendPath = (e) => {
        this.setState({activeDimmer: true});
        axios.post('http://127.0.0.1:8000/process/', {
            path: this.state.path
        })
            .then(res => {
                this.setState({activeDimmer: false, modalOpen: true, image: "data:image/jpeg;base64," + res.data.ret});
            }).catch(error => {
            this.setState({activeDimmer: false});
            alert(error.response.data.error)
        })

    };
    handleClose = () => {
        this.setState({modalOpen: false})
    };
    handle2Close = () => {
        this.setState({modal2Open: false})
    };

    processClick = (e) => {
        let op = null;
        let form = null;
        if (this.getOperation(e.target.id)) {
            op = this.getOperation(e.target.id);
            form = this.createOperationForm(op)
        }
        else {
            op = this.getInput(e.target.id);
            form = this.createInputForm(op);
        }

        this.setState({modalForm: form, modal2Open: true})
    };

    createImageOptions = (img_key) => {
        let options = [];
        let images = this.state.path.nodes;
        for (let image in images) {
            if (img_key !== image) {
                options.push({key: image, text: image, value: image,})
            }
        }
        return options
    };

    deepCopy = (value) => {
        return JSON.parse(JSON.stringify(value));
    };

    createOperationForm = (operation) => {
        this.setState({opImages: this.deepCopy(operation.from), opParams: this.deepCopy(operation.params)});
        let options = this.createImageOptions(operation.to);
        let funct = this.getFunctionByFunction(operation.operation_name);
        let imageForm = (current, options, number) => {
            let forms = [];
            for (let i = 1; i <= number; i++) {
                forms.push(<Form.Select key={"Image " + i} label={"Obraz " + i} defaultValue={current[i - 1]}
                                        name={'image_' + i} options={options}
                                        onChange={this.handleOperationImageChange.bind(this, i - 1)}/>)
            }
            return forms;
        };
        let paramForm = (current, options, number) => {
            let forms = [];
            for (let i = 1; i <= number; i++) {
                forms.push(<Form.Input key={"Image " + i} label={"Parametr " + i}
                                       defaultValue={JSON.stringify(current[i - 1])} name={'param_' + i}
                                       onChange={this.handleOperationParamsChange.bind(this, i - 1)}/>)
            }
            return forms;
        };

        return (
            <Form label={"Edytuj operację:"} onSubmit={this.handleSubmit.bind(this, operation)}>
                <Form.Group>
                    {imageForm(operation.from, options, funct.number_of_images)}
                </Form.Group>
                <Form.Group>
                    {paramForm(operation.params, options, funct.number_of_parameters)}
                </Form.Group>
                <Form.Button basic color='black' content='Edytuj'/>
            </Form>
        )
    };

    createInputForm = (input) => {
        return (
            <Grid>
                <Grid.Row>
                    <Image size={'medium'} name='form' src={"data:image/jpeg;base64," + input.source}/>
                </Grid.Row>
                <Grid.Row>

                    <label className="custom-file-change">
                        <input type='file' accept='image/*' onChange={this.changeImage.bind(this, input)}/>
                        <i className="file image outline icon"/> Załaduj obraz
                    </label>
                </Grid.Row>
            </Grid>


        )
    };

    handleSubmit = (op, e) => {
        let operations = this.state.path.operations;
        for (let operation in operations) {
            if (operations[operation] === op) {
                operations[operation].from = this.state.opImages;
                operations[operation].params = this.state.opParams;
                this.setState(prevState => ({
                    ...prevState,
                    path: {
                        ...prevState.path,
                        operations: operations
                    }
                }));
            }
        }
        this.fixAdjacencies();
        this.handle2Close();
    };

    handleOperationParamsChange = (i, e) => {
        let params = this.state.opParams;
        try {
            params[i] = JSON.parse(e.target.value)
        } catch (ex) {
            params[i] = e.target.value
        }
        this.setState({opParams: params})
    };

    handleOperationImageChange = (i, x, e) => {
        let images = this.state.opImages;
        images[i] = e.value;
        this.setState({opImages: images})
    };

    fixAdjacencies = () => {
        let operations = this.state.path.operations;
        let adjacencies = {};
        for (let opearation in operations) {
            let from = operations[opearation].from;
            for (let source in from) {
                let x = this.deepCopy(from[source]);
                let y = this.deepCopy(operations[opearation].to);
                try {
                    adjacencies[x].push(y);
                }
                catch (e) {
                    adjacencies[x] = [y];
                }
            }
        }
        this.setState(prevState => ({
            ...prevState,
            path: {
                ...prevState.path,
                adjacency: adjacencies
            },
        }), () => {
            this.fixLines();
        });
    };

    fixLines = () => {
        let lines = [];
        let counter = 1;
        let adjacencies = this.state.path.adjacency;
        console.log(this.state.lines);
        for (let adjacency in adjacencies) {
            for (let endNode in adjacencies[adjacency]) {
                let startDiv = this.getDivCenter(adjacency);
                let endDiv = this.getDivCenter(adjacencies[adjacency][endNode]);
                lines[counter - 1] = {
                    name: "line_" + counter,
                    from: adjacency,
                    to: adjacencies[adjacency][endNode],
                    line:
                        <line
                            key={"line_" + counter}
                            x1={startDiv.x}
                            y1={startDiv.y}
                            x2={endDiv.x - 50}
                            y2={endDiv.y - 25}
                            markerEnd={'url(#head)'}
                            style={{
                                stroke: 'black',
                                strokeWidth: 2
                            }}/>
                };
                counter++;
            }
        }
        this.setState({
            lines: lines,
            lineCounter: counter
        })
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
                            style={{padding: 0}}
                        >
                            <FunctionsList
                                disabled={this.state.disabled}
                                active={this.state.activeItem}
                                functions={this.state.functions}
                                handleItemClick={this.handleItemClick}/>
                        </Grid.Column>
                        <Grid.Column
                            width={14}
                            style={{padding: 0}}
                        >
                            <div ref={this.svg_ref} style={{
                                position: 'fixed',
                                zIndex: 1,
                                width: "100%",
                                height: "94vh",
                                overflowY: 'scroll'
                            }}>
                                <svg id={"test"} style={this.state.style}>
                                    <defs>
                                        <marker id='head' orient="auto"
                                                markerWidth='4' markerHeight='8'
                                                refX='0.1' refY='4'>
                                            <path d='M0,0 V8 L4,4 Z' fill="red"/>
                                        </marker>
                                    </defs>
                                    {this.displayLines(this.state.lines)}
                                </svg>

                            </div>
                            <div onScroll={this.handleScroll} onClick={this.addProcess} style={{
                                position: 'fixed',
                                zIndex: 999,
                                width: "100%",
                                height: "94vh",
                                overflowY: 'scroll'
                            }}>
                                {this.displayDivs(this.state.imgs)}
                            </div>
                        </Grid.Column>
                    </Grid>
                </Container>
                <Dimmer active={this.state.activeDimmer}>
                    <Loader indeterminate>Trwa przetwarzanie obrazu</Loader>
                </Dimmer>
                <Modal
                    basic
                    open={this.state.modalOpen}
                    onClose={this.handleClose}
                >
                    <Modal.Content>
                        <Image name='test' src={this.state.image}/>
                    </Modal.Content>
                    <Modal.Actions>
                        <a className="download-button" href={this.state.image} download={'result.jpg'}>
                            <i className="file image outline icon"/> Pobierz obraz
                        </a>
                        <Button color='green' onClick={this.handleClose} inverted>
                            <Icon name='checkmark'/> Zakmnij
                        </Button>
                    </Modal.Actions>
                </Modal>

                <Modal open={this.state.modal2Open} onClose={this.handle2Close}>
                    <Modal.Header>
                        Edycja obrazu
                    </Modal.Header>
                    <Modal.Content>
                        {this.state.modalForm}
                    </Modal.Content>
                    <Modal.Actions>
                        <Button color='green' onClick={this.handle2Close} inverted>
                            <Icon name='checkmark'/> Zamknij
                        </Button>
                    </Modal.Actions>
                </Modal>
            </div>
        );
    }


}

export default App;
