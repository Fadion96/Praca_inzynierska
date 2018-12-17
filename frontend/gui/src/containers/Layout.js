import React from 'react';
import classnames from 'classnames';
import {
    Container,
    Dropdown,
    Menu,
} from 'semantic-ui-react'

const AppMenu = (props) => {
    const appMenuClass = classnames('app_menu');
    return(
            <Menu inverted icon borderless={true} className={appMenuClass}>
                <Container>
                    <Menu.Item as='a' header>
                        Aplikacja przetwarzająca obrazy
                    </Menu.Item>
                    <Menu.Item as='a'>Home</Menu.Item>

                    <Dropdown item simple text='Dropdown'>
                        <Dropdown.Menu>
                            <Dropdown.Item>List Item</Dropdown.Item>
                            <Dropdown.Item>List Item</Dropdown.Item>
                            <Dropdown.Divider />
                            <Dropdown.Header>Header Item</Dropdown.Header>
                            <Dropdown.Item>
                                <i className='dropdown icon' />
                                <span className='text'>Submenu</span>
                                <Dropdown.Menu>
                                    <Dropdown.Item>List Item</Dropdown.Item>
                                    <Dropdown.Item>List Item</Dropdown.Item>
                                </Dropdown.Menu>
                            </Dropdown.Item>
                            <Dropdown.Item>List Item</Dropdown.Item>
                        </Dropdown.Menu>
                    </Dropdown>
                </Container>
            </Menu>
    );
};

export default AppMenu
