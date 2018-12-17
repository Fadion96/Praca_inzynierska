import React from 'react';
import axios from 'axios';
import classname from 'classnames';

import ProcessingFunctions from '../components/ProcessingFunction';

class FunctionsList extends React.Component {




    handleItemClick = (e, { name }) => {
        this.setState({ activeItem: name })
        console.log(name)
    };

    state = {
        functions : [],
        activeItem: null
    };

    componentDidMount() {
        axios.get('http://192.168.0.80:8000/api/')
            .then( res =>{
                this.setState({
                    functions: res.data
                });
                console.log(res.data);
            })
    }

    render() {
        const functionMenuClass = classname('functions-menu');
        return (
            <ProcessingFunctions
                disabled={this.props.disabled}
                className={functionMenuClass}
                functions={this.state.functions}
                onClick={this.handleItemClick}
                activeItem={this.state.activeItem}
            />
        )
    }
}

export default FunctionsList;