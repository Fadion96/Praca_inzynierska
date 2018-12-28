import React from 'react';

import ProcessingFunctions from '../components/ProcessingFunction';

class FunctionsList extends React.Component {


    render() {
        return (
            <ProcessingFunctions
                disabled={this.props.disabled}
                functions={this.props.functions}
                onClick={this.props.handleItemClick}
                activeItem={this.props.active}
            />
        )
    }
}

export default FunctionsList;