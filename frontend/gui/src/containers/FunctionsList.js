import React from 'react';
import axios from 'axios';

import ProcessingFunctions from '../components/ProcessingFunction';

class FunctionsList extends React.Component {

    state = {
        functions : [],
    };

    componentDidMount() {
        axios.get('http://127.0.0.1:8000/api/')
            .then( res =>{
                this.setState({
                    functions: res.data
                });
                console.log(res.data);
            })
    }

    render() {
        return (
            <ProcessingFunctions
                disabled={this.props.disabled}
                functions={this.state.functions}
                onClick={this.props.handleItemClick}
                activeItem={this.props.active}
            />
        )
    }
}

export default FunctionsList;