import React, {Component} from 'react';
import classname from 'classnames'
import './scss/index.scss';
import 'semantic-ui-css/semantic.min.css'
import AppMenu from './components/Layout';
import FunctionsList from './containers/FunctionsList';
import {Button, Container, Dimmer, Form, Grid, Header, Icon, Image, Loader, Modal,} from "semantic-ui-react";
import axios from "axios";
import {saveAs} from 'file-saver';

class App extends Component {
    constructor(props) {
        super(props);
        this.svg_ref = React.createRef();
        this.image_input_ref = React.createRef();
    }

    componentDidMount() {
        axios.get('http://127.0.0.1:8000/process/id')
            .then(res => {
                this.setState({
                    session_id: res.data.id
                });
            }).catch(error => {
            alert(error)
        });
        axios.get('http://127.0.0.1:8000/api/')
            .then(res => {
                this.setState({
                    functions: res.data
                });
            }).catch(error => {
            alert(error)
        })
    }

    state = {
        session_id: null,
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
            inputs: {},
        },
        user_functions: [],
        disabled: true,
        lines: [],
        imgs: [],
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
        let allowedTypes = ["image/png", "image/jpeg", "image/bmp", "image/tiff"];
        if (allowedTypes.includes(file.type)) {
            let fileData = new FileReader();
            fileData.onloadend = (ev) => {
                const content = ev.target.result;
                input.source = content.split(',')[1];
            };
            fileData.readAsDataURL(file);
            this.handle2Close();
        }
        else {
            alert("Niewłaściwy typ obrazka. Wspierane typy: " + allowedTypes)
        }
    };

    uploadFile = (e) => {
        let file = e.target.files[0];
        let formData = new FormData();
        formData.append('file', file);
        formData.append('session_id', this.state.session_id);
        axios.post('http://127.0.0.1:8000/process/upload',
            formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            }
        ).then(res => {
            this.setState(prevState => ({
                ...prevState,
                user_functions: [...prevState.user_functions, res.data.function_id],
                functions: [...prevState.functions, res.data.function],
            }));
        })
            .catch(error => {
                alert(error.response.data.error)
            })
    };

    encodeImage = (e) => {
        let allowedTypes = ["image/png", "image/jpeg", "image/bmp", "image/tiff"];
        let file = e.target.files[0];
        if (allowedTypes.includes(file.type)) {
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
                        top: [(inputCount - 1) * 100 + 25 + "px"]
                    }}>
                    {"Obraz " + inputCount + "\nimg_" + number}
                </div>;
            image.x = 75;
            image.y = (inputCount - 1) * 100 + 55;
            this.setState(prevState => ({
                ...prevState,
                imageCounter: number + 1,
                inputCounter: inputCount + 1,
                imgs: [...prevState.imgs, image],
            }));
            this.image_input_ref.current.value = '';
        }
        else {
            alert("Niewłaściwy typ obrazka. Wspierane typy: " + allowedTypes)
        }
    };

    getDivCenter = (name) => {
        return this.state.imgs.find(el => el.name === name);
    };

    getDivIndex = (name) => {
        return this.state.imgs.findIndex(el => el.name === name);
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

    inInput = (input) => {
        let inputs = this.state.path.inputs;
        for (let inp in inputs) {
            if (inputs[inp] === input) {
                return true;
            }
        }
        return false;
    };

    addProcess = (e) => {
        if (this.state.activeItem != null) {
            let operation = this.getFunction(this.state.activeItem);
            let number = this.state.imageCounter;
            let operationCount = this.state.operationCounter;
            const xPosition = e.pageX - e.target.offsetLeft + e.target.scrollLeft;
            const yPosition = e.pageY - e.target.offsetTop + e.target.scrollTop;
            let number_of_operation_inputs = operation.number_of_images;
            let lastDivKeys = [];
            if (number <= number_of_operation_inputs) {
                let divs = this.state.imgs;
                for (let i = 0; i < number_of_operation_inputs; i++) {
                    lastDivKeys.push(divs[i % (number - 1)].name);
                }
            }
            else {
                let divs = this.state.imgs.slice(-number_of_operation_inputs)
                for (let i = 0; i < number_of_operation_inputs; i++) {
                    lastDivKeys.push(divs[i].name);
                }
            }
            let image = {};
            image.name = "img_" + number;
            image.div =
                <div
                    className={operation.function + "_image_div"}
                    key={["img_" + number]}
                    id={["img_" + number]}
                    onClick={this.processClick}
                    style={{
                        top: [yPosition - 30 + "px"],
                        left: [xPosition - 50 + "px"]
                    }}>
                    {operation.name + "\nimg_" + number}
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
                    nodes: {
                        ...prevState.path.nodes,
                        ["img_" + number]: null
                    },
                    operations: {
                        ...prevState.path.operations,
                        ["operation_" + operationCount]: {
                            operation_name: operation.function,
                            from: lastDivKeys,
                            to: "img_" + number,
                            "params": JSON.parse(operation.params)
                        }
                    }
                },
                disabled: false,
                imgs: [...prevState.imgs, image],
            }), () => {
                this.fixAdjacencies();
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
            path: this.state.path,
            user_functions: this.state.user_functions
        })
            .then(res => {
                this.setState({activeDimmer: false, modalOpen: true, image: "data:image/png;base64," + res.data.ret});
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
            <Form label={"Edytuj operację:"}>
                <Form.Group>
                    {imageForm(operation.from, options, funct.number_of_images)}
                </Form.Group>
                <Form.Group>
                    {paramForm(operation.params, options, funct.number_of_parameters)}
                </Form.Group>
                <Form.Button basic color='black' onClick={this.handleSubmit.bind(this, operation)} content='Edytuj'/>
                <Form.Button basic color='red' onClick={this.deleteImage.bind(this, operation)}
                             content='Usuń operację'/>
            </Form>
        )
    };

    createInputForm = (input) => {
        return (
            <Grid>
                <Grid.Row>
                    <Image bordered size={'medium'} name='form' src={"data:image/png;base64," + input.source}/>
                </Grid.Row>
                <Grid.Row>
                    <label className="custom-file-change">
                        <input type='file' accept='image/*' onChange={this.changeImage.bind(this, input)}/>
                        <i className="file image outline icon"/> Załaduj obraz
                    </label>
                    <Button basic color='red' onClick={this.deleteImage.bind(this, input)} content='Usuń obraz'/>
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

    deletePath = (e) => {
        this.setState({
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
        });
    };

    deleteImage = (operation, e) => {
        let img_key = operation.to;
        if (this.isInAdjacency(img_key)) {
            alert("Nie można usunąć operacji")
        }
        else {
            if (this.inInput(operation)) {
                let inputLength = Object.keys(this.state.path.inputs).length;
                for (let input in this.state.path.inputs) {
                    if (this.state.path.inputs[input] === operation) {
                        let divIndex = this.getDivIndex(img_key);
                        let nodes = this.deleteFromNodes(img_key);
                        let inputs = this.deleteFromInputs(input);
                        let images = this.deleteFromDivs(JSON.stringify(divIndex));
                        if (inputLength === 1) {
                            this.setState(prevState => ({
                                ...prevState,
                                path: {
                                    ...prevState.path,
                                    nodes: nodes,
                                    inputs: inputs,
                                },
                                imgs: images,
                                disabled: true
                            }));
                        } else {
                            this.setState(prevState => ({
                                ...prevState,
                                path: {
                                    ...prevState.path,
                                    nodes: nodes,
                                    inputs: inputs
                                },
                                imgs: images,
                            }));
                        }
                    }
                }
            }
            else {
                for (let op in this.state.path.operations) {
                    if (this.state.path.operations[op] === operation) {
                        let divIndex = this.getDivIndex(img_key);
                        let nodes = this.deleteFromNodes(img_key);
                        let operations = this.deleteFromOperations(op);
                        let images = this.deleteFromDivs(JSON.stringify(divIndex));
                        this.setState(prevState => ({
                            ...prevState,
                            path: {
                                ...prevState.path,
                                nodes: nodes,
                                operations: operations
                            },
                            imgs: images,
                        }), () => {
                            this.fixAdjacencies();
                        });

                    }
                }
            }
        }
        this.handle2Close();
    };

    deleteFromNodes(img_key) {
        let nodes = {};
        for (let node in this.state.path.nodes) {
            if (node !== img_key) {
                nodes[node] = this.deepCopy(this.state.path.nodes[node]);
            }
        }
        return nodes;
    }

    deleteFromInputs(input) {
        let inputs = {};
        for (let inp in this.state.path.inputs) {
            if (inp !== input) {
                inputs[inp] = this.deepCopy(this.state.path.inputs[inp]);
            }
        }
        return inputs;
    }

    deleteFromOperations(operation) {
        let operations = {};
        for (let op in this.state.path.operations) {
            if (op !== operation) {
                operations[op] = this.deepCopy(this.state.path.operations[op]);
            }
        }
        return operations;
    }

    deleteFromDivs(divIndex) {
        let divs = [];
        for (let div in this.state.imgs) {
            if (divIndex !== div) {
                divs.push(this.state.imgs[div]);
            }
        }
        return divs;
    }

    isInAdjacency = (img_key) => {
        let adjacencies = this.state.path.adjacency;
        for (let adjacency in adjacencies) {
            if (adjacency === img_key) {
                return true
            }
        }
        return false
    };

    saveAlgorithm = () => {
        axios.post('http://127.0.0.1:8000/process/check', {
            path: this.state.path,
        })
            .then(res => {
                let inputs = res.data.inputs;
                for (let input in inputs) {
                    inputs[input].source = null;
                }
                const file = new Blob([JSON.stringify(res.data)], {type: 'text/plain'});
                saveAs(file, "algorithm.json")
            })
            .catch(error => {
                alert(error.response.data.error)
            })

    };

    uploadJson = (e) => {
        let file = e.target.files[0];
        let fileData = new FileReader();
        fileData.onloadend = (ev) => {
            const content = ev.target.result;
            let path = JSON.parse(atob(content.split(',')[1]));

            axios.post('http://127.0.0.1:8000/process/check', {
                path: path,
            })
                .then(res => {
                    this.setState(prevState => ({
                        ...prevState,
                        path: this.fixPath(res.data)
                    }), () => {
                        this.createInputs();
                        this.createOperations()
                    });
                })
                .catch(error => {
                    alert(error.response.data.error)
                })

        };
        fileData.readAsDataURL(file);
    };

    fixPath = (path) => {
        let imageCount = 1;
        let inputCount = 1;
        let operationCount = 1;
        let nodes = {};
        let inputs = {};
        let operations = {};
        for (let image in path.nodes) {
            let newKey = "img_" + imageCount;
            if (image === newKey) {
                nodes[newKey] = path.nodes[image];
                for (let inp in path.inputs) {
                    if (image === path.inputs[inp].to) {
                        inputs[("input_" + inputCount)] = path.inputs[inp];
                        delete path.inputs[inp];
                        inputCount++;
                    }
                }
                for (let op in path.operations) {
                    if (image === path.operations[op].to) {
                        operations[("operation_" + operationCount)] = path.operations[op];
                        delete path.operations[op];
                        operationCount++;
                    }
                }
                imageCount++;

            }
            else {
                nodes[newKey] = path.nodes[image];
                for (let inp in path.inputs) {
                    if (image === path.inputs[inp].to) {
                        inputs[("input_" + inputCount)] = path.inputs[inp];
                        inputs[("input_" + inputCount)].to = newKey;
                        for (let op in path.operations) {
                            for (let key in path.operations[op].from) {
                                if (path.operations[op].from[key] === image) {
                                    path.operations[op].from[key] = newKey
                                }
                            }
                        }
                        for (let op in operations) {
                            for (let key in operations[op].from) {
                                if (operations[op].from[key] === image) {
                                    operations[op].from[key] = newKey
                                }
                            }
                        }
                        delete path.inputs[inp];
                        inputCount++;
                    }
                }
                for (let op in path.operations) {
                    if (image === path.operations[op].to) {
                        operations[("operation_" + operationCount)] = path.operations[op];
                        operations[("operation_" + operationCount)].to = newKey;
                        for (let op in path.operations) {
                            for (let key in path.operations[op].from) {
                                if (path.operations[op].from[key] === image) {
                                    path.operations[op].from[key] = newKey
                                }
                            }
                        }
                        for (let op in operations) {
                            for (let key in operations[op].from) {
                                if (operations[op].from[key] === image) {
                                    operations[op].from[key] = newKey
                                }
                            }
                        }
                        delete path.operations[op];
                        operationCount++;
                    }
                }
                imageCount++;
            }
        }
        this.setState({
            imageCounter: imageCount,
            inputCounter: inputCount,
            operationCounter: operationCount
        });
        return {nodes: nodes, inputs: inputs, operations: operations, adjacency: {}}
    };

    createInputs = () => {
        let inputs = this.state.path.inputs;
        let divs = [];
        let number = 1;
        for (let input in inputs) {
            let image = {};
            image.name = inputs[input].to;
            image.div =
                <div
                    className={"input_image_div"}
                    key={inputs[input].to}
                    id={inputs[input].to}
                    onClick={this.processClick}
                    style={{
                        top: [(number - 1) * 100 + 25 + "px"]
                    }}>
                    {"Obraz " + number + "\n" + inputs[input].to}
                </div>;
            image.x = 75;
            image.y = (number - 1) * 100 + 55;
            divs.push(image);
            number++
        }
        this.setState({
            imgs: divs,
            disabled: false
        })
    };

    createOperations = () => {
        let operations = this.state.path.operations;
        let divs = [];
        let number = 1;
        for (let operation in operations) {
            let func = this.getFunctionByFunction(operations[operation].operation_name);
            let image = {};
            image.name = operations[operation].to;
            image.div =
                <div
                    className={func.function + "_image_div"}
                    key={operations[operation].to}
                    id={operations[operation].to}
                    onClick={this.processClick}
                    style={{
                        top: [(number - 1) * 100 + 25 + "px"],
                        left: [225 + "px"]
                    }}>
                    {func.name + "\n" + operations[operation].to}
                </div>;
            image.x = 275;
            image.y = (number - 1) * 100 + 55;
            divs.push(image);
            number++
        }
        this.setState(prevState => ({
            ...prevState,
            imgs: [...prevState.imgs, ...divs],
            disabled: false
        }), () => {
            this.fixAdjacencies();
        });

    };

    imageToBlob = (image) => {
        let bytes = atob(image.split(',')[1]);
        let type = image.split(',')[0].split(':')[1].split(';')[0];
        let buffer = new ArrayBuffer(bytes.length);
        let intArray = new Uint8Array(buffer);
        for (let i = 0; i < bytes.length; i++) {
            intArray[i] = bytes.charCodeAt(i);
        }
        return new Blob([buffer], {type: type})
    };


    downloadFile = () => {
        const file = this.imageToBlob(this.state.image);
        saveAs(file, "result.png")
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
                        handleDelete={this.deletePath}
                        fileHandler={this.uploadFile}
                        saveAlgorithm={this.saveAlgorithm}
                        jsonHandler={this.uploadJson}
                        inputImageRef={this.image_input_ref}
                        disabled={this.state.disabled}
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
                            <Dimmer.Dimmable dimmed={this.state.disabled}>
                                <FunctionsList
                                    disabled={this.state.disabled}
                                    active={this.state.activeItem}
                                    functions={this.state.functions}
                                    handleItemClick={this.handleItemClick}/>
                            </Dimmer.Dimmable>
                            <Dimmer active={this.state.disabled}>
                                <Header as='h2' icon inverted>
                                    <Icon name='upload'/>
                                </Header>
                                <Header as='h5' inverted>
                                    Załaduj obraz, aby odblokować menu
                                </Header>
                            </Dimmer>
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
                        <Button className="download-button" onClick={this.downloadFile} inverted>
                            <i className="file image outline icon"/> Pobierz obraz
                        </Button>
                        <Button color='green' onClick={this.handleClose} inverted>
                            <Icon name='checkmark'/> Zamknij
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